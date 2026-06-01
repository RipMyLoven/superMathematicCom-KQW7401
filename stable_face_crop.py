import math
import torch
import numpy as np
import comfy.utils
from comfy.model_management import get_torch_device

# ─── Общий детектор лиц ───

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
        print("[LipsyncCrop] Using insightface detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    try:
        import mediapipe as mp
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        DETECTOR = face_detection
        DETECTOR_TYPE = "mediapipe"
        print("[LipsyncCrop] Using mediapipe detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        DETECTOR = cv2.CascadeClassifier(cascade_path)
        DETECTOR_TYPE = "opencv"
        print("[LipsyncCrop] Using OpenCV Haar cascade detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    raise RuntimeError("[LipsyncCrop] No face detector available!")


def detect_face_bbox(frame_np_uint8, detector, detector_type):
    h, w = frame_np_uint8.shape[:2]
    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        faces = detector.get(bgr)
        if not faces:
            return None
        biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = biggest.bbox.astype(int)
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    elif detector_type == "mediapipe":
        results = detector.process(frame_np_uint8)
        if not results.detections:
            return None
        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w); y2 = int((bb.ymin + bb.height) * h)
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        biggest = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = biggest
        return (x, y, x + fw, y + fh)
    return None


def detect_face_full(frame_np_uint8, detector, detector_type):
    h, w = frame_np_uint8.shape[:2]
    result = {'bbox': None, 'landmarks': {}, 'all_found': False, 'face_ratio': 0.0}
    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        faces = detector.get(bgr)
        if not faces:
            return result
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        result['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        result['face_ratio'] = ((x2 - x1) * (y2 - y1)) / (w * h)
        if face.kps is not None and len(face.kps) >= 5:
            kps = face.kps.astype(int)
            result['landmarks']['left_eye'] = tuple(kps[0])
            result['landmarks']['right_eye'] = tuple(kps[1])
            result['landmarks']['nose'] = tuple(kps[2])
            result['landmarks']['mouth_left'] = tuple(kps[3])
            result['landmarks']['mouth_right'] = tuple(kps[4])
            result['all_found'] = True
    elif detector_type == "mediapipe":
        results = detector.process(frame_np_uint8)
        if not results.detections:
            return result
        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w); y2 = int((bb.ymin + bb.height) * h)
        result['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        result['face_ratio'] = ((x2 - x1) * (y2 - y1)) / (w * h)
        kp = det.location_data.relative_keypoints
        if len(kp) >= 4:
            result['landmarks']['right_eye'] = (int(kp[0].x * w), int(kp[0].y * h))
            result['landmarks']['left_eye'] = (int(kp[1].x * w), int(kp[1].y * h))
            result['landmarks']['nose'] = (int(kp[2].x * w), int(kp[2].y * h))
            mx = int(kp[3].x * w); my = int(kp[3].y * h)
            ed = abs(result['landmarks']['left_eye'][0] - result['landmarks']['right_eye'][0])
            hm = max(ed // 3, 10)
            result['landmarks']['mouth_left'] = (mx - hm, my)
            result['landmarks']['mouth_right'] = (mx + hm, my)
            result['all_found'] = True
    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return result
        b = max(faces, key=lambda f: f[2] * f[3])
        result['bbox'] = (b[0], b[1], b[0] + b[2], b[1] + b[3])
        result['face_ratio'] = (b[2] * b[3]) / (w * h)
    return result


# ═══════════════════════════════════════════════════════════════
# НОДА 1: Lipsync (ручной режим, квадратный выход)
# ═══════════════════════════════════════════════════════════════

class LipsyncCrop:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "smoothing": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA сглаживание. Выше = плавнее. 0.8-0.9 оптимально"
                }),
                "window_size": ("INT", {
                    "default": 7, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Окно скользящего среднего (нечётное). 5-11 для 30fps"
                }),
                "scale_padding": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Множитель области вокруг лица. 1.5 = 50% запас"
                }),
                "shift_vertical": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "<0.5 = больше лба, >0.5 = больше подбородка"
                }),
                "output_size": ("INT", {
                    "default": 512, "min": 128, "max": 1024, "step": 64,
                    "tooltip": "Размер выходного квадратного кропа"
                }),
                "size_stabilization": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Макс. отклонение размера от ��едианы. 0.1 = ±10%"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Детектить лицо каждый N-й кадр"
                }),
                "resolution_divider": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Делитель разрешения. 1.0=полное, 2.0=÷2 (быстрее upscale). Округляет до ×8"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_video",)
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Стабильная вырезка лица для lipsync. Ручной scale_padding, квадратный выход."

    def process(self, images, smoothing=0.85, window_size=7, scale_padding=1.5,
                shift_vertical=0.45, output_size=512, size_stabilization=0.1,
                detect_every_n=1, resolution_divider=1.0):

        B, H, W, C = images.shape
        final_size = max(64, (int(output_size / resolution_divider) // 8) * 8)

        print(f"[Lipsync] {B} frames ({W}x{H}) → {final_size}x{final_size} "
              f"(base={output_size}, ÷{resolution_divider:.2f})")

        detector, detector_type = get_face_detector()
        pbar = comfy.utils.ProgressBar(B)

        raw_bboxes = []
        for i in range(B):
            if i % detect_every_n == 0:
                frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                bbox = detect_face_bbox(frame_np, detector, detector_type)
                raw_bboxes.append(bbox)
            else:
                raw_bboxes.append(None)
            pbar.update_absolute(i, B)

        valid_indices = [i for i, b in enumerate(raw_bboxes) if b is not None]
        if len(valid_indices) == 0:
            print("[Lipsync] No face detected! Center crop.")
            return (self._center_crop_batch(images, final_size, final_size),)

        filled_cx = np.zeros(B, dtype=np.float64)
        filled_cy = np.zeros(B, dtype=np.float64)
        filled_s = np.zeros(B, dtype=np.float64)

        for i in range(B):
            if raw_bboxes[i] is not None:
                x1, y1, x2, y2 = raw_bboxes[i]
                face_w = x2 - x1; face_h = y2 - y1
                area = face_w * face_h * scale_padding
                side = math.sqrt(area)
                filled_cx[i] = (x1 + x2) / 2.0
                filled_cy[i] = (y1 + y2) / 2.0 + side * (0.5 - shift_vertical) * 0.3
                filled_s[i] = side
            else:
                filled_cx[i] = np.nan; filled_cy[i] = np.nan; filled_s[i] = np.nan

        filled_cx = _interpolate_nans(filled_cx)
        filled_cy = _interpolate_nans(filled_cy)
        filled_s = _interpolate_nans(filled_s)

        smooth_cx = _bidirectional_ema(filled_cx, smoothing)
        smooth_cy = _bidirectional_ema(filled_cy, smoothing)
        smooth_s = _bidirectional_ema(filled_s, smoothing)

        if window_size > 1:
            smooth_cx = _moving_average(smooth_cx, window_size)
            smooth_cy = _moving_average(smooth_cy, window_size)
            smooth_s = _moving_average(smooth_s, window_size)

        if size_stabilization > 0:
            median_s = np.median(smooth_s)
            smooth_s = np.clip(smooth_s,
                               median_s * (1.0 - size_stabilization),
                               median_s * (1.0 + size_stabilization))

        result_frames = []
        for i in range(B):
            frame = images[i]
            cx, cy, s = smooth_cx[i], smooth_cy[i], smooth_s[i]
            half = s / 2.0
            crop_x1 = cx - half; crop_y1 = cy - half
            crop_x2 = cx + half; crop_y2 = cy + half
            if crop_x1 < 0: crop_x2 -= crop_x1; crop_x1 = 0
            if crop_y1 < 0: crop_y2 -= crop_y1; crop_y1 = 0
            if crop_x2 > W: crop_x1 -= (crop_x2 - W); crop_x2 = W
            if crop_y2 > H: crop_y1 -= (crop_y2 - H); crop_y2 = H
            crop_x1 = int(max(0, crop_x1)); crop_y1 = int(max(0, crop_y1))
            crop_x2 = int(min(W, crop_x2)); crop_y2 = int(min(H, crop_y2))
            if crop_x2 - crop_x1 < 10 or crop_y2 - crop_y1 < 10:
                crop_x1 = max(0, int(cx - 50)); crop_y1 = max(0, int(cy - 50))
                crop_x2 = min(W, crop_x1 + 100); crop_y2 = min(H, crop_y1 + 100)
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
            cropped = cropped.unsqueeze(0).permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(
                cropped, size=(final_size, final_size), mode='bilinear', align_corners=False)
            result_frames.append(resized.squeeze(0).permute(1, 2, 0))

        result = torch.stack(result_frames, dim=0)
        print(f"[Lipsync] Done. {result.shape}")
        return (result,)

    def _center_crop_batch(self, images, out_w, out_h):
        B, H, W, C = images.shape
        ar = out_w / out_h
        if W / H > ar: crop_h = H; crop_w = int(H * ar)
        else: crop_w = W; crop_h = int(W / ar)
        y1 = (H - crop_h) // 2; x1 = (W - crop_w) // 2
        c = images[:, y1:y1+crop_h, x1:x1+crop_w, :].permute(0, 3, 1, 2)
        r = torch.nn.functional.interpolate(c, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return r.permute(0, 2, 3, 1)


# ═════════════════════════════��═════════════════════════════════
# НОДА 2: Lipsync AUTO (авто-масштаб по landmarks, произвольный AR)
# ═══════════════════════════════════════════════════════════════

class LipsyncAutoCrop:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Ширина выхода. 512×512=квадрат, 720×1280=портрет"
                }),
                "output_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Высота выхода"
                }),
                "resolution_divider": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Делитель разрешения. 1.0=полное, 2.0=÷2 (быстрее upscale). Округляет до ×8"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "Сглаживание позиции. Выше = плавнее"
                }),
                "window_size": ("INT", {
                    "default": 5, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Окно скользящего среднего"
                }),
                "shift_vertical": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "<0.5 = больше лба, >0.5 = больше подбородка"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Детектить каждый N-й кадр"
                }),
                # ── Эти два параметра рядом внизу ──
                "auto_scale_padding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ВКЛ = авто-масштаб по landmarks (scale_padding игнорируется). ВЫКЛ = ручной scale_padding"
                }),
                "scale_padding": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Ручной множитель (работает ТОЛЬКО если auto_scale_padding ВЫКЛ)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_video",)
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Авто-масштаб по landmarks ИЛИ ручной scale_padding. Произвольный AR. Делитель разрешения."

    def _clamp_crop(self, cx, cy, crop_w, crop_h, frame_w, frame_h):
        ar = crop_w / crop_h if crop_h > 0 else 1.0
        if crop_w > frame_w:
            crop_w = float(frame_w); crop_h = crop_w / ar
        if crop_h > frame_h:
            crop_h = float(frame_h); crop_w = crop_h * ar
            if crop_w > frame_w:
                crop_w = float(frame_w); crop_h = crop_w / ar
        half_w = crop_w / 2.0; half_h = crop_h / 2.0
        if cx - half_w < 0: cx = half_w
        if cx + half_w > frame_w: cx = frame_w - half_w
        if cy - half_h < 0: cy = half_h
        if cy + half_h > frame_h: cy = frame_h - half_h
        return cx, cy, crop_w, crop_h

    def _get_crop_for_frame_auto(self, face_info, frame_w, frame_h, shift_vertical, aspect_ratio):
        """Авто-масштаб по landmarks."""
        bbox = face_info['bbox']
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        bw = x2 - x1; bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        landmarks = face_info.get('landmarks', {})
        all_found = face_info.get('all_found', False)
        bbox_cx = (x1 + x2) / 2.0; bbox_cy = (y1 + y2) / 2.0

        if not all_found:
            face_size = max(bw, bh) * 2.0
            if aspect_ratio >= 1.0: crop_w = face_size * aspect_ratio; crop_h = face_size
            else: crop_w = face_size; crop_h = face_size / aspect_ratio
            cy = bbox_cy - crop_h * (shift_vertical - 0.5) * 0.4
            return (bbox_cx, cy, crop_w, crop_h)

        left_eye = landmarks.get('left_eye'); right_eye = landmarks.get('right_eye')
        mouth_l = landmarks.get('mouth_left'); mouth_r = landmarks.get('mouth_right')
        pts = [(n, p) for n, p in landmarks.items() if p is not None]
        all_x = [p[1][0] for p in pts]

        if len(pts) < 3:
            face_size = max(bw, bh) * 2.0
            if aspect_ratio >= 1.0: crop_w = face_size * aspect_ratio; crop_h = face_size
            else: crop_w = face_size; crop_h = face_size / aspect_ratio
            cy = bbox_cy - crop_h * (shift_vertical - 0.5) * 0.4
            return (bbox_cx, cy, crop_w, crop_h)

        eye_dist = 0.0; eye_center_x = bbox_cx; eye_center_y = bbox_cy
        if left_eye and right_eye:
            eye_dist = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
            eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
            eye_center_y = (left_eye[1] + right_eye[1]) / 2.0

        face_vert = 0.0; mouth_center_y = bbox_cy
        if left_eye and right_eye and mouth_l and mouth_r:
            mouth_center_y = (mouth_l[1] + mouth_r[1]) / 2.0
            face_vert = abs(mouth_center_y - eye_center_y)

        if face_vert > 5:
            top_of_head = eye_center_y - face_vert * 1.20
            bottom_of_chin = mouth_center_y + face_vert * 0.90
        elif eye_dist > 5:
            top_of_head = eye_center_y - eye_dist * 1.4
            bottom_of_chin = eye_center_y + eye_dist * 2.5
        else:
            top_of_head = y1 - bh * 0.5; bottom_of_chin = y2 + bh * 0.3

        if eye_dist > 5:
            left_of_face = min(all_x) - eye_dist * 0.65
            right_of_face = max(all_x) + eye_dist * 0.65
        else:
            left_of_face = x1 - bw * 0.35; right_of_face = x2 + bw * 0.35

        real_face_w = right_of_face - left_of_face
        real_face_h = bottom_of_chin - top_of_head
        real_face_cx = (left_of_face + right_of_face) / 2.0
        real_face_cy = (top_of_head + bottom_of_chin) / 2.0

        need_w = real_face_w * 1.6; need_h = real_face_h * 1.6
        if need_w / aspect_ratio >= need_h:
            crop_w = need_w; crop_h = crop_w / aspect_ratio
        else:
            crop_h = need_h; crop_w = crop_h * aspect_ratio

        cx = real_face_cx; cy = real_face_cy

        check_points = list(pts)
        check_points.append(("head", (real_face_cx, top_of_head)))
        check_points.append(("chin", (real_face_cx, bottom_of_chin)))
        check_points.append(("faceL", (left_of_face, real_face_cy)))
        check_points.append(("faceR", (right_of_face, real_face_cy)))

        for _ in range(10):
            ok = True
            half_w = crop_w / 2.0; half_h = crop_h / 2.0
            margin_x = crop_w * 0.18; margin_y = crop_h * 0.18
            for name, (px, py) in check_points:
                dl = px - (cx - half_w); dr = (cx + half_w) - px
                dt = py - (cy - half_h); db = (cy + half_h) - py
                if dl < margin_x:
                    crop_w += (margin_x - dl) * 2; crop_h = crop_w / aspect_ratio; ok = False
                if dr < margin_x:
                    crop_w += (margin_x - dr) * 2; crop_h = crop_w / aspect_ratio; ok = False
                if dt < margin_y:
                    crop_h += (margin_y - dt) * 2; crop_w = crop_h * aspect_ratio; ok = False
                if db < margin_y:
                    crop_h += (margin_y - db) * 2; crop_w = crop_h * aspect_ratio; ok = False
            if ok:
                break

        cy = cy - crop_h * (shift_vertical - 0.5) * 0.35
        crop_w = max(crop_w, bw); crop_h = max(crop_h, bh)
        return (cx, cy, float(crop_w), float(crop_h))

    def _get_crop_for_frame_manual(self, bbox, scale_padding, shift_vertical, aspect_ratio, frame_w, frame_h):
        """Ручной scale_padding (без landmarks)."""
        if bbox is None:
            return None
        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1; bh = by2 - by1
        if bw <= 0 or bh <= 0:
            return None

        face_size = max(bw, bh) * scale_padding
        if aspect_ratio >= 1.0:
            cw = face_size * aspect_ratio; ch = face_size
        else:
            cw = face_size; ch = face_size / aspect_ratio

        ccx = (bx1 + bx2) / 2.0
        ccy = (by1 + by2) / 2.0
        ccy -= ch * (shift_vertical - 0.5) * 0.4
        return (ccx, ccy, float(cw), float(ch))

    def _crop_frame(self, frame, cx, cy, crop_w, crop_h, out_w, out_h):
        H, W, C = frame.shape
        x1 = int(max(0, round(cx - crop_w / 2.0)))
        y1 = int(max(0, round(cy - crop_h / 2.0)))
        x2 = int(min(W, round(cx + crop_w / 2.0)))
        y2 = int(min(H, round(cy + crop_h / 2.0)))
        if x2 - x1 < 4: x1 = max(0, x2 - 4)
        if y2 - y1 < 4: y1 = max(0, y2 - 4)
        cropped = frame[y1:y2, x1:x2, :]
        cropped = cropped.unsqueeze(0).permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            cropped, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)

    def process(self, images, output_width=512, output_height=512,
                smoothing=0.7, window_size=5, shift_vertical=0.45,
                detect_every_n=1, auto_scale_padding=True, scale_padding=1.5,
                resolution_divider=1.0):

        B, H, W, C = images.shape
        aspect_ratio = output_width / output_height

        final_w = max(64, (int(output_width / resolution_divider) // 8) * 8)
        final_h = max(64, (int(output_height / resolution_divider) // 8) * 8)

        mode_str = "AUTO (landmarks)" if auto_scale_padding else f"MANUAL (scale={scale_padding})"
        print(f"[Lipsync AUTO] {B} frames ({W}x{H}) → {final_w}x{final_h} "
              f"(base={output_width}x{output_height}, ÷{resolution_divider:.2f}, "
              f"AR={aspect_ratio:.2f}, mode={mode_str})")

        if not auto_scale_padding:
            print(f"[Lipsync AUTO] scale_padding={scale_padding} (manual mode)")
        else:
            print(f"[Lipsync AUTO] scale_padding IGNORED (auto mode)")

        detector, detector_type = get_face_detector()
        pbar = comfy.utils.ProgressBar(B)

        raw_cx = np.full(B, np.nan); raw_cy = np.full(B, np.nan)
        raw_cw = np.full(B, np.nan); raw_ch = np.full(B, np.nan)

        for i in range(B):
            if i % detect_every_n == 0:
                frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                info = detect_face_full(frame_np, detector, detector_type)

                if auto_scale_padding:
                    # ── АВТО: landmarks определяют масштаб ──
                    crop = self._get_crop_for_frame_auto(
                        info, W, H, shift_vertical, aspect_ratio)
                else:
                    # ── РУЧНОЙ: scale_padding определяет масштаб ──
                    crop = self._get_crop_for_frame_manual(
                        info['bbox'], scale_padding, shift_vertical, aspect_ratio, W, H)

                if crop:
                    raw_cx[i], raw_cy[i] = crop[0], crop[1]
                    raw_cw[i], raw_ch[i] = crop[2], crop[3]
                    if i < 5 or i % 50 == 0:
                        fr = info.get('face_ratio', 0)
                        print(f"  [f{i}] ratio={fr:.3f} "
                              f"crop={crop[2]:.0f}x{crop[3]:.0f} frame={W}x{H}")

            pbar.update_absolute(i, B)

        if not np.any(~np.isnan(raw_cx)):
            print("[Lipsync AUTO] No face! Center crop.")
            return (self._center_crop_batch(images, final_w, final_h),)

        raw_cx = _interpolate_nans(raw_cx); raw_cy = _interpolate_nans(raw_cy)
        raw_cw = _interpolate_nans(raw_cw); raw_ch = _interpolate_nans(raw_ch)

        smooth_cx = _bidirectional_ema(raw_cx, smoothing)
        smooth_cy = _bidirectional_ema(raw_cy, smoothing)
        if window_size > 1:
            smooth_cx = _moving_average(smooth_cx, window_size)
            smooth_cy = _moving_average(smooth_cy, window_size)

        smooth_cw = _moving_average(raw_cw, 3)
        smooth_ch = _moving_average(raw_ch, 3)

        print(f"[Lipsync AUTO] Crop range: "
              f"w={smooth_cw.min():.0f}-{smooth_cw.max():.0f} "
              f"h={smooth_ch.min():.0f}-{smooth_ch.max():.0f}")

        result_frames = []
        for i in range(B):
            clamped_cx, clamped_cy, cw_i, ch_i = self._clamp_crop(
                smooth_cx[i], smooth_cy[i], smooth_cw[i], smooth_ch[i], W, H)
            cropped = self._crop_frame(
                images[i], clamped_cx, clamped_cy, cw_i, ch_i, final_w, final_h)
            result_frames.append(cropped)

        result = torch.stack(result_frames, dim=0)
        print(f"[Lipsync AUTO] Done. {result.shape}")
        return (result,)

    def _center_crop_batch(self, images, out_w, out_h):
        B, H, W, C = images.shape
        ar = out_w / out_h
        if W / H > ar: crop_h = H; crop_w = int(H * ar)
        else: crop_w = W; crop_h = int(W / ar)
        y1 = (H - crop_h) // 2; x1 = (W - crop_w) // 2
        c = images[:, y1:y1+crop_h, x1:x1+crop_w, :].permute(0, 3, 1, 2)
        r = torch.nn.functional.interpolate(c, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return r.permute(0, 2, 3, 1)


# ═══════════════════════════════════════════════════════════════
# Общие утилиты
# ═══════════════════════════════════════════════════════════════

def _interpolate_nans(arr):
    nans = np.isnan(arr)
    if not np.any(nans): return arr
    if np.all(nans): return np.zeros_like(arr)
    valid = ~nans; idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr

def _bidirectional_ema(values, alpha):
    n = len(values)
    if n <= 1: return values.copy()
    fwd = np.zeros(n); fwd[0] = values[0]
    for i in range(1, n): fwd[i] = alpha * fwd[i-1] + (1-alpha) * values[i]
    bwd = np.zeros(n); bwd[-1] = values[-1]
    for i in range(n-2, -1, -1): bwd[i] = alpha * bwd[i+1] + (1-alpha) * values[i]
    return (fwd + bwd) / 2.0

def _moving_average(values, window):
    if window <= 1: return values
    hw = window // 2
    padded = np.pad(values, hw, mode='reflect')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# ═══════════════════════════════════════════════════════════════
# НОДА 3: Stable Face Crop GOD — топовая стабилизация
# ═══════════════════════════════════════════════════════════════
#
# Что внутри:
#   • Детект лица + landmarks (eye-line) → cx, cy, size, angle
#   • Optical-flow fallback (phase correlation) если детектор молчит —
#     трекинг продолжается, кадр не теряется.
#   • One Euro filter — адаптивная плавность: мелкая дрожь убирается,
#     быстрые движения сохраняются.
#   • Velocity / zoom limit — клампит максимальный сдвиг и зум за кадр,
#     никаких "прыжков".
#   • IQR-clamp размера + deadzone положения — стабильный масштаб.
#   • Bidirectional EMA + Savitzky-Golay polish.
#   • Single-pass warpAffine: rotate + zoom + translate за одну операцию,
#     быстро и без артефактов (BORDER_REFLECT — никаких чёрных рамок).
#   • Полностью fault-tolerant: даже если детектор сдох на всех кадрах
#     — фоллбэк на центр-кроп, не падает.
# ═══════════════════════════════════════════════════════════════

class StableFaceCropGod:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Ширина выхода"
                }),
                "output_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Высота выхода"
                }),
                "scale_padding": ("FLOAT", {
                    "default": 1.8, "min": 1.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Множитель области вокруг лица"
                }),
                "shift_vertical": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "<0.5 = больше лба, >0.5 = больше подбородка"
                }),
                "shift_horizontal": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0.5 = центр, <0.5 = сдвиг влево, >0.5 = вправо"
                }),
                "enable_rotation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Авто-выравнивание лица по линии глаз"
                }),
                "max_rotation_deg": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Макс. угол поворота (клампится)"
                }),
                "rotation_smoothing": ("FLOAT", {
                    "default": 0.92, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA для угла. Выше = плавнее поворот"
                }),
                "position_smoothing": ("FLOAT", {
                    "default": 0.88, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA базовая для cx/cy"
                }),
                "size_smoothing": ("FLOAT", {
                    "default": 0.96, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA для размера/зума. Выше = меньше дыхания"
                }),
                "lock_size_to_median": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ЖЁСТКО фиксирует размер кропа на медиане — нулевое зум-дыхание, чистый слайд квадрата."
                }),
                "pre_median_window": ("INT", {
                    "default": 5, "min": 0, "max": 31, "step": 1,
                    "tooltip": "Медианный фильтр ДО сглаживания (убивает одиночные выбросы детектора). 0=выкл, нечётное."
                }),
                "one_euro_min_cutoff": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "One Euro filter: ниже = плавнее в покое"
                }),
                "one_euro_beta": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "One Euro β: выше = резвее на быстрых движениях"
                }),
                "deadzone_pixels": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Игнор движений меньше N px (анти-дрожь)"
                }),
                "max_velocity_px": ("FLOAT", {
                    "default": 120.0, "min": 0.0, "max": 1000.0, "step": 1.0,
                    "tooltip": "Макс. сдвиг центра кропа за кадр (0=выкл). Высокое = успевает за быстрой камерой"
                }),
                "motion_lookahead": ("INT", {
                    "default": 3, "min": 0, "max": 15, "step": 1,
                    "tooltip": "Смотреть на N кадров вперёд (предсказание движения). 0=выкл, 2-5 = успевает за быстрой камерой"
                }),
                "safety_containment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ГАРАНТИЯ что raw bbox лица всегда внутри финального кропа — расширяет size если нужно. Спасает от обрезания на резких движениях"
                }),
                "safety_margin": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Запас вокруг raw bbox при safety_containment (доля от размера)"
                }),
                "motion_adaptive_padding": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Расширение кропа пропорционально скорости движения. 0=выкл, 0.25 = +25% размера на быстрых движениях"
                }),
                "max_zoom_per_frame": ("FLOAT", {
                    "default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Макс. относ. изменение размера за кадр (0=выкл)"
                }),
                "size_iqr_clamp": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Клампит размер вокруг медианы ±N (0=выкл)"
                }),
                "savgol_window": ("INT", {
                    "default": 9, "min": 0, "max": 51, "step": 1,
                    "tooltip": "Savitzky-Golay окно (0=выкл, нечётное)"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Детектить каждый N-й кадр"
                }),
                "optical_flow_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Если детектор не нашёл лицо — трекать через phase correlation"
                }),
                "border_mode": (["reflect", "replicate", "black", "wrap"], {
                    "default": "reflect",
                    "tooltip": "Чем заполнять если кроп вышел за кадр"
                }),
                "resolution_divider": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Делитель разрешения (1.0=полное)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_video",)
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "🔥 GOD-TIER stable face crop: rotation align, zoom checks, optical-flow fallback, One Euro filter, velocity limits."

    # ── вспомогалки ──────────────────────────────────────────────

    @staticmethod
    def _angle_from_eyes(le, re):
        if le is None or re is None:
            return 0.0
        dx = re[0] - le[0]; dy = re[1] - le[1]
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _phase_shift(prev_gray, cur_gray):
        """Phase correlation → (dx, dy) сдвига. None если не вышло."""
        try:
            import cv2
            if prev_gray is None or cur_gray is None:
                return None
            if prev_gray.shape != cur_gray.shape:
                return None
            (dx, dy), _ = cv2.phaseCorrelate(
                prev_gray.astype(np.float32),
                cur_gray.astype(np.float32))
            return float(dx), float(dy)
        except Exception:
            return None

    @staticmethod
    def _safe_gray_patch(frame_u8, cx, cy, size, target=128):
        """Берёт квадратный патч вокруг (cx,cy) и ресайзит до target."""
        try:
            import cv2
            H, W = frame_u8.shape[:2]
            half = max(8, int(size / 2))
            x1 = max(0, int(cx) - half); y1 = max(0, int(cy) - half)
            x2 = min(W, int(cx) + half); y2 = min(H, int(cy) + half)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return None
            patch = frame_u8[y1:y2, x1:x2]
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            return cv2.resize(gray, (target, target), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

    def process(self, images,
                output_width=512, output_height=512,
                scale_padding=1.8, shift_vertical=0.45, shift_horizontal=0.5,
                enable_rotation=False, max_rotation_deg=25.0, rotation_smoothing=0.92,
                position_smoothing=0.88, size_smoothing=0.96,
                lock_size_to_median=True, pre_median_window=5,
                one_euro_min_cutoff=1.0, one_euro_beta=0.02,
                deadzone_pixels=3.0, max_velocity_px=120.0, max_zoom_per_frame=0.04,
                size_iqr_clamp=0.20, savgol_window=9,
                detect_every_n=1, optical_flow_fallback=True,
                border_mode="reflect", resolution_divider=1.0,
                motion_lookahead=3, safety_containment=True, safety_margin=0.10,
                motion_adaptive_padding=0.25):

        import cv2
        B, H, W, C = images.shape
        final_w = max(64, (int(output_width / resolution_divider) // 8) * 8)
        final_h = max(64, (int(output_height / resolution_divider) // 8) * 8)
        aspect = output_width / max(1, output_height)

        print(f"[GOD] {B} frames ({W}x{H}) → {final_w}x{final_h} "
              f"(AR={aspect:.2f}, rot={enable_rotation}, flow={optical_flow_fallback})")

        # Детектор — мягко, не падаем
        try:
            detector, detector_type = get_face_detector()
        except Exception as e:
            print(f"[GOD] Detector failed: {e}. Center crop fallback.")
            return (self._center_crop_batch(images, final_w, final_h),)

        pbar = comfy.utils.ProgressBar(B)

        raw_cx = np.full(B, np.nan)
        raw_cy = np.full(B, np.nan)
        raw_size = np.full(B, np.nan)
        raw_angle = np.full(B, np.nan)
        detected_mask = np.zeros(B, dtype=bool)

        # Кэш патчей для phase correlation
        prev_patch = None
        prev_anchor = None  # (cx, cy, size)
        frames_u8 = [None] * B  # ленивый кэш

        def _frame_u8(i):
            if frames_u8[i] is None:
                frames_u8[i] = (images[i].cpu().numpy() * 255).astype(np.uint8)
            return frames_u8[i]

        # ── ПРОХОД 1: детекция + flow fallback ──
        for i in range(B):
            do_detect = (i % detect_every_n == 0)
            cx = cy = size = angle = None

            if do_detect:
                try:
                    info = detect_face_full(_frame_u8(i), detector, detector_type)
                except Exception:
                    info = {'bbox': None, 'landmarks': {}, 'all_found': False}

                bbox = info.get('bbox')
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    bw = x2 - x1; bh = y2 - y1
                    if bw > 0 and bh > 0:
                        face_size = max(bw, bh) * scale_padding
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        size = float(face_size)
                        lm = info.get('landmarks', {})
                        if enable_rotation:
                            angle = self._angle_from_eyes(
                                lm.get('left_eye'), lm.get('right_eye'))
                            if angle is not None:
                                angle = float(np.clip(angle, -max_rotation_deg, max_rotation_deg))
                        else:
                            angle = 0.0

            # Fallback по optical flow
            if cx is None and optical_flow_fallback and prev_patch is not None and prev_anchor is not None:
                pcx, pcy, psize = prev_anchor
                cur_patch = self._safe_gray_patch(_frame_u8(i), pcx, pcy, psize, target=128)
                shift = self._phase_shift(prev_patch, cur_patch)
                if shift is not None and cur_patch is not None:
                    # phaseCorrelate возвращает сдвиг в координатах патча (128x128),
                    # масштабируем обратно к размеру исходного патча.
                    scale_back = psize / 128.0
                    dx_full = shift[0] * scale_back
                    dy_full = shift[1] * scale_back
                    # Клампим, чтобы flow не утащил трек в космос
                    lim = psize * 0.3
                    dx_full = float(np.clip(dx_full, -lim, lim))
                    dy_full = float(np.clip(dy_full, -lim, lim))
                    cx = pcx + dx_full
                    cy = pcy + dy_full
                    size = psize
                    # angle остаётся None → NaN → интерполируется из соседей,
                    # чтобы не обнулять стабильный поворот лица.

            if cx is not None:
                raw_cx[i] = cx
                raw_cy[i] = cy
                raw_size[i] = size
                # angle: если из детекции — пишем; если из flow — оставляем NaN
                if angle is not None:
                    raw_angle[i] = angle
                detected_mask[i] = True
                # Обновляем кэш для следующего flow
                prev_anchor = (cx, cy, size)
                prev_patch = self._safe_gray_patch(_frame_u8(i), cx, cy, size, target=128)

            pbar.update_absolute(i, B)

        n_det = int(detected_mask.sum())
        print(f"[GOD] Tracked {n_det}/{B} frames "
              f"({100.0*n_det/max(1,B):.1f}%)")

        # Полный провал — центр-кроп, никаких падений
        if n_det == 0:
            print("[GOD] No tracking at all. Center crop fallback.")
            return (self._center_crop_batch(images, final_w, final_h),)

        # ── ПРОХОД 2: интерполяция NaN ──
        raw_cx = _interpolate_nans(raw_cx)
        raw_cy = _interpolate_nans(raw_cy)
        raw_size = _interpolate_nans(raw_size)
        if enable_rotation:
            # Если детектор вообще не отдал ни одного угла — все нули.
            if np.all(np.isnan(raw_angle)):
                raw_angle = np.zeros(B)
            else:
                raw_angle = _interpolate_nans(raw_angle)
        else:
            raw_angle = np.zeros(B)

        # ── ПРОХОД 2.5: медианный пре-фильтр (убивает выбросы) ──
        if pre_median_window and pre_median_window >= 3:
            mw = pre_median_window if pre_median_window % 2 == 1 else pre_median_window + 1
            raw_cx = _median_filter(raw_cx, mw)
            raw_cy = _median_filter(raw_cy, mw)
            raw_size = _median_filter(raw_size, mw)
            if enable_rotation:
                raw_angle = _median_filter(raw_angle, mw)

        # ── ПРОХОД 3: IQR clamp размера (анти-выбросы) ──
        if size_iqr_clamp > 0 and n_det >= 4:
            med = float(np.median(raw_size))
            lo = med * (1.0 - size_iqr_clamp)
            hi = med * (1.0 + size_iqr_clamp)
            raw_size = np.clip(raw_size, lo, hi)

        # ── LOCK SIZE: жёстко фиксируем размер на медиане ──
        # Самый радикальный анти-дёрг — кадр становится чистым слайдом квадрата.
        if lock_size_to_median:
            locked = float(np.median(raw_size))
            raw_size = np.full(B, locked, dtype=np.float64)
            print(f"[GOD] 🔒 Size LOCKED to median = {locked:.1f}px (zero zoom breathing)")

        # ── ПРОХОД 4: One Euro filter (адаптивная плавность) ──
        sm_cx = _one_euro(raw_cx, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
        sm_cy = _one_euro(raw_cy, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
        sm_size = _one_euro(raw_size, min_cutoff=one_euro_min_cutoff * 0.7, beta=one_euro_beta * 0.5)
        sm_angle = _one_euro(raw_angle, min_cutoff=one_euro_min_cutoff * 0.5, beta=one_euro_beta * 0.3) if enable_rotation else raw_angle

        # ── ПРОХОД 5: Bidirectional EMA ──
        sm_cx = _bidirectional_ema(sm_cx, position_smoothing)
        sm_cy = _bidirectional_ema(sm_cy, position_smoothing)
        sm_size = _bidirectional_ema(sm_size, size_smoothing)
        if enable_rotation:
            sm_angle = _bidirectional_ema(sm_angle, rotation_smoothing)

        # ── ПРОХОД 6: Savitzky-Golay (опционально) ──
        if savgol_window and savgol_window >= 5:
            w = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
            sm_cx = _savgol(sm_cx, w, 3)
            sm_cy = _savgol(sm_cy, w, 3)
            sm_size = _savgol(sm_size, w, 2)
            if enable_rotation:
                sm_angle = _savgol(sm_angle, w, 2)

        # ── ПРОХОД 7: Deadzone (анти-дрожь) ──
        if deadzone_pixels > 0:
            sm_cx = _deadzone(sm_cx, deadzone_pixels)
            sm_cy = _deadzone(sm_cy, deadzone_pixels)

        # ── ПРОХОД 8: velocity / zoom limit ──
        if max_velocity_px > 0:
            sm_cx = _velocity_clamp(sm_cx, max_velocity_px)
            sm_cy = _velocity_clamp(sm_cy, max_velocity_px)
        if max_zoom_per_frame > 0 and not lock_size_to_median:
            sm_size = _relative_velocity_clamp(sm_size, max_zoom_per_frame)
        if enable_rotation and max_velocity_px > 0:
            # для угла лимит грубо — 5° за кадр
            sm_angle = _velocity_clamp(sm_angle, 5.0)

        # Если lock_size включён — никакого One Euro / Savgol для size,
        # держим строго медианное значение.
        if lock_size_to_median:
            sm_size = np.full(B, np.median(raw_size), dtype=np.float64)

        # ── ПРОХОД 8.5: MOTION LOOKAHEAD — центр смотрит вперёд ──
        # На быстрой камере сглаженный центр отстаёт от реального лица.
        # Сдвигаем center в направлении будущего движения.
        if motion_lookahead > 0 and B > motion_lookahead + 1:
            la = int(motion_lookahead)
            future_cx = np.empty(B)
            future_cy = np.empty(B)
            for i in range(B):
                j = min(B - 1, i + la)
                future_cx[i] = raw_cx[j]
                future_cy[i] = raw_cy[j]
            # Бленд: 70% сглаженного + 30% будущего
            blend = 0.30
            sm_cx = sm_cx * (1.0 - blend) + future_cx * blend
            sm_cy = sm_cy * (1.0 - blend) + future_cy * blend
            print(f"[GOD] Lookahead {la} frames активен — центр смотрит вперёд")

        # ── ПРОХОД 8.6: MOTION-ADAPTIVE PADDING ──
        # На быстрых движениях расширяем кроп пропорционально скорости,
        # чтобы лицо не вылетало из кадра.
        if motion_adaptive_padding > 0 and B >= 2:
            vx = np.gradient(sm_cx)
            vy = np.gradient(sm_cy)
            speed = np.hypot(vx, vy)  # px/кадр
            # Нормируем по медианному размеру: скорость в "размерах лица за кадр"
            ref_size = float(np.median(sm_size))
            speed_rel = speed / max(1.0, ref_size)  # типично 0.0..0.3
            # Расширяем size: при speed_rel=0.2 → +motion_adaptive_padding * ratio
            expand = 1.0 + motion_adaptive_padding * np.clip(speed_rel * 5.0, 0.0, 1.0)
            # Сглаживаем expand чтобы не дёргался
            expand = _bidirectional_ema(expand, 0.85)
            sm_size = sm_size * expand
            max_exp = float(expand.max())
            if max_exp > 1.02:
                print(f"[GOD] Motion-adaptive padding: max expand ×{max_exp:.2f} на быстрых кадрах")

        # ── ПРОХОД 8.7: SAFETY CONTAINMENT ──
        # Гарантия что raw bbox лица всегда влезает в финальный кроп.
        # Если smoothed crop отстал — расширяем size, чтобы вместить.
        if safety_containment and aspect >= 1.0:
            crop_w_arr = sm_size * aspect
            crop_h_arr = sm_size.copy()
        else:
            crop_w_arr = sm_size.copy()
            crop_h_arr = sm_size / max(1e-6, aspect)

        if safety_containment:
            margin = 1.0 + safety_margin
            need_size = np.zeros(B)
            for i in range(B):
                if not detected_mask[i]:
                    need_size[i] = sm_size[i]
                    continue
                # raw_cx/cy/size — оригинальная позиция + размер лица
                # Нужно чтобы [raw_cx ± raw_size*aspect/2, raw_cy ± raw_size/2] был
                # внутри [sm_cx ± crop_w/2, sm_cy ± crop_h/2].
                rcx, rcy, rs = raw_cx[i], raw_cy[i], raw_size[i] * margin
                if aspect >= 1.0:
                    rcw = rs * aspect; rch = rs
                else:
                    rcw = rs; rch = rs / aspect
                # Необходимая ширина кропа чтобы влез bbox с учётом смещения центра
                dx = abs(rcx - sm_cx[i]); dy = abs(rcy - sm_cy[i])
                need_w = 2.0 * (dx + rcw / 2.0)
                need_h = 2.0 * (dy + rch / 2.0)
                if aspect >= 1.0:
                    s_from_w = need_w / aspect
                    s_from_h = need_h
                else:
                    s_from_w = need_w
                    s_from_h = need_h * aspect
                need_size[i] = max(s_from_w, s_from_h)
            # Расширяем только если нужно (никогда не уменьшаем)
            expand_safety = np.maximum(sm_size, need_size)
            # Лёгкое сглаживание чтобы расширения не моргали
            expand_safety = _bidirectional_ema(expand_safety, 0.7)
            # Защита от паразитного роста
            expand_safety = np.minimum(expand_safety, sm_size * 2.5)
            grew = (expand_safety > sm_size * 1.01).sum()
            if grew > 0:
                print(f"[GOD] Safety containment: расширил кроп на {grew} кадрах (быстрые движения)")
            sm_size = expand_safety

        # Применяем shift_horizontal/vertical на центр
        # (сдвигаем "куда смотреть" внутри кропа)
        # shift=0.5 — без сдвига; иначе двигаем центр к нужной стороне
        # величина сдвига пропорциональна размеру кропа
        h_shift_arr = (shift_horizontal - 0.5) * sm_size * 0.4
        v_shift_arr = (shift_vertical - 0.5) * sm_size * 0.4
        sm_cx_shifted = sm_cx + h_shift_arr
        sm_cy_shifted = sm_cy - v_shift_arr  # vert: <0.5 = больше сверху → центр вниз

        print(f"[GOD] size range: {sm_size.min():.0f}-{sm_size.max():.0f} "
              f"(median {np.median(sm_size):.0f})")
        if enable_rotation:
            a_span = sm_angle.max() - sm_angle.min()
            print(f"[GOD] angle range: {sm_angle.min():.1f}° - {sm_angle.max():.1f}° (span {a_span:.1f}°)")
            if a_span < 0.5:
                print(f"[GOD] ⚠ Rotation ~0°: лицо в кадре уже ровное либо детектор не отдал landmarks.")

        # ── РЕНДЕР: warpAffine = rotate + scale + translate за раз ──
        border = {
            "reflect":   cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "black":     cv2.BORDER_CONSTANT,
            "wrap":      cv2.BORDER_WRAP,
        }.get(border_mode, cv2.BORDER_REFLECT_101)

        result_frames = []
        # ширина кропа в исходных пикселях пропорциональна aspect
        for i in range(B):
            cx = float(sm_cx_shifted[i]); cy = float(sm_cy_shifted[i])
            size = float(max(8.0, sm_size[i]))
            ang = float(sm_angle[i]) if enable_rotation else 0.0

            # Прямоугольник кропа в исходнике:
            if aspect >= 1.0:
                crop_w = size * aspect; crop_h = size
            else:
                crop_w = size; crop_h = size / aspect

            # Scale: исходный crop_w → final_w
            scale = final_w / crop_w

            try:
                frame_u8 = _frame_u8(i)
                # Композиция: 1) translate центр кропа в (0,0)
                #             2) rotate на -ang (выравниваем лицо)
                #             3) scale
                #             4) translate в центр final
                a = math.radians(-ang) if enable_rotation else 0.0
                ca = math.cos(a) * scale
                sa = math.sin(a) * scale
                # affine matrix: dst = M * [x,y,1]
                M = np.array([
                    [ca, -sa, final_w / 2.0 - (ca * cx - sa * cy)],
                    [sa,  ca, final_h / 2.0 - (sa * cx + ca * cy)],
                ], dtype=np.float32)
                warped = cv2.warpAffine(
                    frame_u8, M, (final_w, final_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=border,
                    borderValue=(0, 0, 0))
                t = torch.from_numpy(warped.astype(np.float32) / 255.0)
                result_frames.append(t)
            except Exception as e:
                # Никогда не падаем на одном кадре
                print(f"[GOD] frame {i} warp failed: {e}, using fallback crop")
                fallback = self._fallback_crop(images[i], cx, cy, crop_w, crop_h, final_w, final_h)
                result_frames.append(fallback)

        result = torch.stack(result_frames, dim=0).to(images.device)
        print(f"[GOD] Done. {result.shape}")
        return (result,)

    @staticmethod
    def _fallback_crop(frame, cx, cy, cw, ch, out_w, out_h):
        H, W, C = frame.shape
        x1 = int(max(0, round(cx - cw / 2.0)))
        y1 = int(max(0, round(cy - ch / 2.0)))
        x2 = int(min(W, round(cx + cw / 2.0)))
        y2 = int(min(H, round(cy + ch / 2.0)))
        if x2 - x1 < 4: x1 = max(0, x2 - 4)
        if y2 - y1 < 4: y1 = max(0, y2 - 4)
        cropped = frame[y1:y2, x1:x2, :].unsqueeze(0).permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            cropped, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)

    def _center_crop_batch(self, images, out_w, out_h):
        B, H, W, C = images.shape
        ar = out_w / out_h
        if W / H > ar: crop_h = H; crop_w = int(H * ar)
        else: crop_w = W; crop_h = int(W / ar)
        y1 = (H - crop_h) // 2; x1 = (W - crop_w) // 2
        c = images[:, y1:y1+crop_h, x1:x1+crop_w, :].permute(0, 3, 1, 2)
        r = torch.nn.functional.interpolate(c, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return r.permute(0, 2, 3, 1)


# ═══════════════════════════════════════════════════════════════
# Дополнительные утилиты для GOD-режима
# ═══════════════════════════════════════════════════════════════

def _one_euro(values, min_cutoff=1.0, beta=0.02, d_cutoff=1.0, freq=30.0):
    """One Euro filter — адаптивная плавность.
       Низкая скорость → сильно сглаживаем (cutoff низкий).
       Высокая скорость → меньше сглаживаем (cutoff растёт)."""
    n = len(values)
    if n == 0:
        return values
    out = np.zeros(n)
    out[0] = values[0]
    prev_v = values[0]
    prev_dv = 0.0
    dt = 1.0 / max(1e-6, freq)

    def alpha(cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    for i in range(1, n):
        v = values[i]
        dv = (v - prev_v) / dt
        a_d = alpha(d_cutoff)
        dv_hat = a_d * dv + (1 - a_d) * prev_dv
        cutoff = min_cutoff + beta * abs(dv_hat)
        a = alpha(cutoff)
        v_hat = a * v + (1 - a) * prev_v
        out[i] = v_hat
        prev_v = v_hat
        prev_dv = dv_hat
    return out


def _savgol(values, window, polyorder):
    """Savitzky-Golay фильтр (плавная полиномиальная аппроксимация)."""
    try:
        from scipy.signal import savgol_filter
        if len(values) < window:
            return values
        return savgol_filter(values, window, min(polyorder, window - 1))
    except Exception:
        # Fallback на moving average
        return _moving_average(values, window)


def _deadzone(values, threshold):
    """Игнорирует движения меньше threshold пикселей."""
    out = values.copy()
    for i in range(1, len(out)):
        if abs(out[i] - out[i-1]) < threshold:
            out[i] = out[i-1]
    return out


def _velocity_clamp(values, max_delta):
    """Лимитирует максимальный сдвиг за кадр."""
    out = values.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i-1]
        if abs(delta) > max_delta:
            out[i] = out[i-1] + np.sign(delta) * max_delta
    return out


def _relative_velocity_clamp(values, max_rel):
    """Лимитирует относительное изменение (для size/zoom)."""
    out = values.copy()
    for i in range(1, len(out)):
        prev = out[i-1]
        if prev <= 0:
            continue
        rel = (out[i] - prev) / prev
        if abs(rel) > max_rel:
            out[i] = prev * (1.0 + np.sign(rel) * max_rel)
    return out


def _median_filter(values, window):
    """Скользящий медианный фильтр — убивает одиночные выбросы
       (детектор моргнул и выдал ерунду на 1 кадр)."""
    if window <= 1 or len(values) < window:
        return values
    try:
        from scipy.signal import medfilt
        return medfilt(values, kernel_size=window)
    except Exception:
        hw = window // 2
        padded = np.pad(values, hw, mode='edge')
        out = np.empty_like(values)
        for i in range(len(values)):
            out[i] = np.median(padded[i:i + window])
        return out


# ═══════════════════════════════════════════════════════════════
# Регистрация нод
# ═══════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "LipsyncCrop": LipsyncCrop,
    "LipsyncAutoCrop": LipsyncAutoCrop,
    "StableFaceCropGod": StableFaceCropGod,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipsyncCrop": "🎯 Lipsync",
    "LipsyncAutoCrop": "🎯 Lipsync AUTO",
    "StableFaceCropGod": "🔥 Stable Face Crop GOD",
}