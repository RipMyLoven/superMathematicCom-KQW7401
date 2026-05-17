# comfyui-mouth-only-stable.py
import math
import torch
import numpy as np
import comfy.utils

DETECTOR = None
DETECTOR_TYPE = None

def get_face_detector():
    global DETECTOR, DETECTOR_TYPE
    if DETECTOR is not None:
        return DETECTOR, DETECTOR_TYPE
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        DETECTOR = app
        DETECTOR_TYPE = "insightface"
        print("[MouthOnly] Using insightface detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    try:
        import mediapipe as mp
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        DETECTOR = face_detection
        DETECTOR_TYPE = "mediapipe"
        print("[MouthOnly] Using mediapipe detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    raise RuntimeError("[MouthOnly] No face detector available!")

def detect_face_full(frame_np_uint8, detector, detector_type):
    h, w = frame_np_uint8.shape[:2]
    result = {'bbox': None, 'landmarks': {}, 'all_found': False}
    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        faces = detector.get(bgr)
        if not faces:
            return result
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1,y1,x2,y2 = face.bbox.astype(int)
        result['bbox'] = (max(0,x1), max(0,y1), min(w,x2), min(h,y2))
        if face.kps is not None and len(face.kps)>=5:
            kps = face.kps.astype(int)
            result['landmarks']['mouth_left'] = tuple(kps[3])
            result['landmarks']['mouth_right'] = tuple(kps[4])
            result['all_found'] = True
    elif detector_type == "mediapipe":
        results = detector.process(frame_np_uint8)
        if not results.detections:
            return result
        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin*w); y1=int(bb.ymin*h)
        x2=int((bb.xmin+bb.width)*w); y2=int((bb.ymin+bb.height)*h)
        result['bbox'] = (max(0,x1), max(0,y1), min(w,x2), min(h,y2))
        kp = det.location_data.relative_keypoints
        if len(kp)>=4:
            mx=int(kp[3].x*w); my=int(kp[3].y*h)
            ed=abs(int(kp[1].x*w)-int(kp[0].x*w))
            hm=max(ed//3,10)
            result['landmarks']['mouth_left']=(mx-hm,my)
            result['landmarks']['mouth_right']=(mx+hm,my)
            result['all_found']=True
    return result

def _interpolate_nans(arr):
    nans = np.isnan(arr)
    if not np.any(nans): return arr
    if np.all(nans): return np.zeros_like(arr)
    valid = ~nans; idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr

def smooth_with_limit(arr, alpha=0.85, max_jump=0.15):
    smoothed = arr.copy()
    for i in range(1, len(arr)):
        prev = smoothed[i-1]
        curr = arr[i]
        if not np.isnan(prev) and not np.isnan(curr):
            max_delta = prev * max_jump
            delta = curr - prev
            if abs(delta) > max_delta:
                curr = prev + np.sign(delta) * max_delta
        smoothed[i] = alpha*prev + (1-alpha)*curr
    return smoothed

def get_auto_mouth_crop(face_info, W, H, min_size=32, max_size=512, extra_pad=1.4):
    bbox = face_info.get('bbox', None)
    landmarks = face_info.get('landmarks', {})
    all_found = face_info.get('all_found', False)
    if not all_found or 'mouth_left' not in landmarks or 'mouth_right' not in landmarks:
        cx, cy = W/2, H/2
        size = min(W,H)/4
        return cx, cy, size
    ml = np.array(landmarks['mouth_left'])
    mr = np.array(landmarks['mouth_right'])
    mouth_cx = (ml[0]+mr[0])/2
    mouth_cy = (ml[1]+mr[1])/2
    mouth_w = np.linalg.norm(mr-ml)
    if bbox is not None:
        _, y1, _, y2 = bbox
        mouth_h = max(mouth_w*0.8, (y2 - mouth_cy)*1.2)
    else:
        mouth_h = mouth_w
    # Применяем extra_pad отдельно
    pad_w = mouth_w * extra_pad
    pad_h = mouth_h * extra_pad
    crop_size = max(pad_w, pad_h)
    crop_size = np.clip(crop_size, min_size, max_size*2)  # временно без сильного обрезания
    return mouth_cx, mouth_cy, crop_size

class MouthOnlyCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_size": ("INT", {"default":128, "min":32, "max":1024, "step":8}),
                "detect_every_n": ("INT", {"default":1, "min":1, "max":30, "step":1}),
                "smoothing": ("FLOAT", {"default":0.85, "min":0.0, "max":0.99, "step":0.01}),
                "extra_pad": ("FLOAT", {"default":1.4, "min":1.0, "max":5.0, "step":0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mouth_video",)
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Вырезка рта с авто-паддингом: всегда губы и язык в кадре, без лишнего тела."

    def process(self, images, output_size=128, detect_every_n=1,
                smoothing=0.85, extra_pad=1.4):
        B, H, W, C = images.shape
        final_size = max(32, (output_size//8)*8)
        detector, detector_type = get_face_detector()
        pbar = comfy.utils.ProgressBar(B)

        cx_arr = np.full(B, np.nan)
        cy_arr = np.full(B, np.nan)
        size_arr = np.full(B, np.nan)

        for i in range(B):
            if i % detect_every_n == 0:
                frame_np = (images[i].cpu().numpy()*255).astype(np.uint8)
                face_info = detect_face_full(frame_np, detector, detector_type)
                cx, cy, size = get_auto_mouth_crop(face_info, W, H, min_size=32, max_size=output_size, extra_pad=extra_pad)
                cx_arr[i] = cx
                cy_arr[i] = cy
                size_arr[i] = size
            pbar.update_absolute(i, B)

        # Интерполяция пропавших детекций
        cx_raw = _interpolate_nans(cx_arr)
        cy_raw = _interpolate_nans(cy_arr)
        size_raw = _interpolate_nans(size_arr)

        # Сглаживание с лимитом скачков
        cx_smooth = smooth_with_limit(cx_raw, alpha=smoothing, max_jump=0.15)
        cy_smooth = smooth_with_limit(cy_raw, alpha=smoothing, max_jump=0.15)
        size_smooth = smooth_with_limit(size_raw, alpha=smoothing, max_jump=0.15)

        result_frames = []
        for i in range(B):
            cx, cy, s = cx_smooth[i], cy_smooth[i], size_smooth[i]
            half = s / 2

            # авто-центрирование: не даём кропу выйти за границы кадра
            cx = np.clip(cx, half, W - half)
            cy = np.clip(cy, half, H - half)

            x1 = int(cx - half)
            y1 = int(cy - half)
            x2 = int(cx + half)
            y2 = int(cy + half)

            # минимальный размер fallback
            if x2-x1 < 4 or y2-y1 < 4:
                x1, y1 = 0, 0
                x2, y2 = min(W, final_size), min(H, final_size)

            cropped = images[i][y1:y2, x1:x2, :].unsqueeze(0).permute(0,3,1,2)
            resized = torch.nn.functional.interpolate(
                cropped, size=(final_size, final_size), mode='bilinear', align_corners=False)
            result_frames.append(resized.squeeze(0).permute(1,2,0))

        result = torch.stack(result_frames, dim=0)
        print(f"[MouthOnlyCrop] Done. {result.shape}")
        return (result,)

NODE_CLASS_MAPPINGS = {
    "MouthOnlyCrop": MouthOnlyCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MouthOnlyCrop": "👄 Mouth Only Crop",
}