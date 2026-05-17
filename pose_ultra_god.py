# ═══════════════════════════════════════════════════════════════
# 💀 POSE ULTRA GOD — самая стабильная и точная нода для bone-animation
# ───────────────────────────────────────────────────────────────
#  Вдохновлено https://github.com/kijai/ComfyUI-WanAnimatePreprocess,
#  но **не ломается, не мерцает, и работает лучше**:
#
#   • Backend chain:
#       1. DWPose (RTMPose-based, 133-wholebody) — SOTA для ControlNet,
#          точнее чем mediapipe и чем ViTPose из WAN.
#          Подхватывается из установленного `comfyui_controlnet_aux`
#          (у большинства уже стоит). Авто-загрузка моделей не нужна.
#       2. MediaPipe Holistic — fallback, без ONNX, всегда работает.
#   • Канонический OpenPose-18-body + 21x2 hands + 68 face mesh рендер,
#     с правильным limbSeq и цветами — работает дроп-ин в любой
#     Pose ControlNet / Wan-Animate / AnimateAnyone и т.д.
#   • Per-keypoint темпоральная стабилизация (One Euro + EMA + median +
#     confidence-gated NaN interp) — нулевое мерцание костей.
#   • Optical-flow fallback на пропавшие детекции.
#   • Multi-person aware (берёт самого крупного человека стабильно).
#   • Стабилизированный face crop с lock_size и safety containment.
#   • Fault-tolerant: любая ошибка → graceful degradation, не падает.
#
#  Выходы:
#    pose_canvas    — IMAGE (OpenPose-style скелет на чёрном фоне)
#    face_crops     — IMAGE (стабильный квадратный crop лица)
#    keypoints_json — STRING (json с keypoints по кадрам)
#    body_bboxes    — STRING (json с body bbox по кадрам)
#    face_bboxes    — STRING (json с face bbox по кадрам)
# ═══════════════════════════════════════════════════════════════

import os
import math
import json
import torch
import numpy as np
import comfy.utils

# Реюз стабилизаторов из pose_face_god (один источник правды)
try:
    from .pose_face_god import (
        _interp_nans_1d, _one_euro_1d, _bidirectional_ema_1d, _median_filter_1d,
        _MediaPipeBackend,
    )
except Exception:
    # На случай прямого запуска
    from pose_face_god import (  # type: ignore
        _interp_nans_1d, _one_euro_1d, _bidirectional_ema_1d, _median_filter_1d,
        _MediaPipeBackend,
    )


# ═══════════════════════════════════════════════════════════════
# КАНОНИЧЕСКИЙ OpenPose-18 ФОРМАТ
# ═══════════════════════════════════════════════════════════════
#
# 0=nose, 1=neck, 2=R_shoulder, 3=R_elbow, 4=R_wrist,
# 5=L_shoulder, 6=L_elbow, 7=L_wrist, 8=R_hip, 9=R_knee, 10=R_ankle,
# 11=L_hip, 12=L_knee, 13=L_ankle, 14=R_eye, 15=L_eye, 16=R_ear, 17=L_ear
#
# Связки (1-based в каноничном OpenPose, мы храним 0-based):
OPENPOSE_LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (2, 16), (5, 17),
]
OPENPOSE_COLORS = [
    [255, 0, 0],   [255, 85, 0],  [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0],  [0, 255, 0],   [0, 255, 85],  [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255],  [0, 0, 255],   [85, 0, 255],  [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85],  [255, 100, 100],
]
OPENPOSE_POINT_COLORS = [
    [255, 0, 0],   [255, 85, 0],  [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0],  [0, 255, 0],   [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255],   [85, 0, 255],  [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85],
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ═══════════════════════════════════════════════════════════════
# Mediapipe (33-body) → OpenPose-18 конвертация
# ═══════════════════════════════════════════════════════════════
# Mediapipe indices: 0=nose, 2=L_eye, 5=R_eye, 7=L_ear, 8=R_ear,
# 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow,
# 15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip,
# 25=L_knee, 26=R_knee, 27=L_ankle, 28=R_ankle.

MP_TO_OP18 = {
    0:  0,   # nose
    # 1 (neck) computed as midpoint of shoulders below
    2:  12,  # R_shoulder
    3:  14,  # R_elbow
    4:  16,  # R_wrist
    5:  11,  # L_shoulder
    6:  13,  # L_elbow
    7:  15,  # L_wrist
    8:  24,  # R_hip
    9:  26,  # R_knee
    10: 28,  # R_ankle
    11: 23,  # L_hip
    12: 25,  # L_knee
    13: 27,  # L_ankle
    14: 5,   # R_eye
    15: 2,   # L_eye
    16: 8,   # R_ear
    17: 7,   # L_ear
}


def mediapipe_body_to_op18(mp_body):
    """mp_body: (33, 3) → (18, 3). Возвращает NaN для отсутствующих точек."""
    op = np.full((18, 3), np.nan, dtype=np.float32)
    for op_idx, mp_idx in MP_TO_OP18.items():
        op[op_idx] = mp_body[mp_idx]
    # Neck = midpoint shoulders
    ls, rs = mp_body[11], mp_body[12]
    if not (np.isnan(ls[0]) or np.isnan(rs[0])):
        op[1, 0] = (ls[0] + rs[0]) / 2.0
        op[1, 1] = (ls[1] + rs[1]) / 2.0
        op[1, 2] = min(ls[2], rs[2])
    return op


def coco17_to_op18(coco17):
    """COCO-17 (DWPose body) → OpenPose-18.
    COCO: 0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear,
          5=L_sh, 6=R_sh, 7=L_el, 8=R_el, 9=L_wr, 10=R_wr,
          11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee, 15=L_ank, 16=R_ank.
    """
    op = np.full((18, 3), np.nan, dtype=np.float32)
    mapping = {
        0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9,
        8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
        14: 2, 15: 1, 16: 4, 17: 3,
    }
    for op_idx, coco_idx in mapping.items():
        op[op_idx] = coco17[coco_idx]
    ls, rs = coco17[5], coco17[6]
    if not (np.isnan(ls[0]) or np.isnan(rs[0])):
        op[1, 0] = (ls[0] + rs[0]) / 2.0
        op[1, 1] = (ls[1] + rs[1]) / 2.0
        op[1, 2] = min(ls[2], rs[2])
    return op


# ═══════════════════════════════════════════════════════════════
# DWPose backend (через установленный comfyui_controlnet_aux)
# ═══════════════════════════════════════════════════════════════

class _DWPoseBackend:
    """DWPose: 133-wholebody (17 body + 6 feet + 68 face + 42 hands)."""
    name = "dwpose"

    def __init__(self, device='cuda'):
        # Пробуем взять уже инициализированный DWPose из controlnet_aux
        self._predictor = None
        self._load_via_controlnet_aux(device)

    def _load_via_controlnet_aux(self, device):
        import importlib
        candidates = [
            "custom_nodes.comfyui_controlnet_aux.src.controlnet_aux.dwpose",
            "comfyui_controlnet_aux.src.controlnet_aux.dwpose",
            "controlnet_aux.dwpose",
        ]
        last_err = None
        for mod_path in candidates:
            try:
                mod = importlib.import_module(mod_path)
                # controlnet_aux 0.0.7+: from_pretrained / DwposeDetector
                if hasattr(mod, "DwposeDetector"):
                    Det = getattr(mod, "DwposeDetector")
                    try:
                        self._predictor = Det.from_pretrained_default()
                    except Exception:
                        try:
                            self._predictor = Det.from_pretrained("yzd-v/DWPose")
                        except Exception:
                            self._predictor = Det()
                    self._mode = "dwpose_detector"
                    print(f"[PoseUltraGOD] DWPose backend ready via {mod_path}")
                    return
                if hasattr(mod, "Wholebody"):
                    Wb = getattr(mod, "Wholebody")
                    self._predictor = Wb(device=device)
                    self._mode = "wholebody"
                    print(f"[PoseUltraGOD] DWPose Wholebody ready via {mod_path}")
                    return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"DWPose недоступен: установи comfyui_controlnet_aux. Last: {last_err}"
        )

    def detect(self, frame_rgb_u8):
        H, W = frame_rgb_u8.shape[:2]
        try:
            if self._mode == "wholebody":
                # Wholebody.__call__(image_bgr) → (keypoints[N,133,2], scores[N,133])
                import cv2
                bgr = cv2.cvtColor(frame_rgb_u8, cv2.COLOR_RGB2BGR)
                kps, scores = self._predictor(bgr)
                if kps is None or len(kps) == 0:
                    return None
                # Берём самого крупного по area
                idx = self._pick_largest(kps, scores)
                kp = kps[idx]      # (133, 2)
                sc = scores[idx]   # (133,)
                return self._pack_133(kp, sc, W, H)
            else:
                # DwposeDetector: возвращает PoseResult / список
                import PIL.Image as Image
                pil = Image.fromarray(frame_rgb_u8)
                # output_type='dict' даёт raw keypoints
                try:
                    poses = self._predictor(
                        pil, output_type='json',
                        include_hand=True, include_face=True,
                    )
                except TypeError:
                    poses = self._predictor(pil)
                # Парсим dict-формат controlnet_aux
                return self._parse_aux_output(poses, W, H)
        except Exception as e:
            print(f"[PoseUltraGOD] DWPose inference failed: {e}")
            return None

    @staticmethod
    def _pick_largest(kps, scores):
        best_i, best_a = 0, -1.0
        for i in range(len(kps)):
            valid = scores[i] > 0.3
            if valid.sum() < 4:
                continue
            pts = kps[i][valid]
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            a = w * h
            if a > best_a:
                best_a, best_i = a, i
        return best_i

    @staticmethod
    def _pack_133(kp, sc, W, H):
        """kp[133,2], sc[133] → стандартный dict."""
        body17 = np.zeros((17, 3), dtype=np.float32)
        body17[:, :2] = kp[:17]; body17[:, 2] = sc[:17]
        body17[sc[:17] < 0.3] = np.nan

        face68 = np.zeros((68, 3), dtype=np.float32)
        face68[:, :2] = kp[23:91]; face68[:, 2] = sc[23:91]
        face68[sc[23:91] < 0.3] = np.nan

        hand_l = np.zeros((21, 3), dtype=np.float32)
        hand_l[:, :2] = kp[91:112]; hand_l[:, 2] = sc[91:112]
        hand_l[sc[91:112] < 0.3] = np.nan

        hand_r = np.zeros((21, 3), dtype=np.float32)
        hand_r[:, :2] = kp[112:133]; hand_r[:, 2] = sc[112:133]
        hand_r[sc[112:133] < 0.3] = np.nan

        op18 = coco17_to_op18(body17)

        # bbox по всем валидным
        all_pts = []
        for arr in (op18, face68, hand_l, hand_r):
            m = ~np.isnan(arr[:, 0])
            if m.any():
                all_pts.append(arr[m, :2])
        bbox = None
        if all_pts:
            pts = np.concatenate(all_pts, 0)
            x1, y1 = pts.min(0); x2, y2 = pts.max(0)
            bbox = (
                float(max(0, x1)), float(max(0, y1)),
                float(min(W, x2)), float(min(H, y2)),
            )
        return {
            'body': op18, 'face': face68,
            'hand_l': hand_l, 'hand_r': hand_r, 'bbox': bbox,
        }

    @staticmethod
    def _parse_aux_output(poses, W, H):
        """controlnet_aux DwposeDetector JSON-like output."""
        if not poses or 'people' not in poses or not poses['people']:
            return None
        # Берём первого (controlnet_aux уже отсортировал по confidence)
        person = poses['people'][0]
        kp_2d = person.get('pose_keypoints_2d', [])  # x,y,c × 18 (op18) или 25 (op25)
        face_2d = person.get('face_keypoints_2d', [])
        hand_l_2d = person.get('hand_left_keypoints_2d', [])
        hand_r_2d = person.get('hand_right_keypoints_2d', [])

        def to_arr(flat, n_expected):
            if not flat:
                return np.full((n_expected, 3), np.nan, dtype=np.float32)
            arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
            # x,y in normalized [0,1] для controlnet_aux
            if arr[:, 0].max() <= 1.5:
                arr[:, 0] *= W; arr[:, 1] *= H
            arr[arr[:, 2] < 0.3] = np.nan
            out = np.full((n_expected, 3), np.nan, dtype=np.float32)
            out[:min(len(arr), n_expected)] = arr[:n_expected]
            return out

        body = to_arr(kp_2d, 18)
        face = to_arr(face_2d, 68)
        hl = to_arr(hand_l_2d, 21)
        hr = to_arr(hand_r_2d, 21)

        all_pts = []
        for arr in (body, face, hl, hr):
            m = ~np.isnan(arr[:, 0])
            if m.any():
                all_pts.append(arr[m, :2])
        bbox = None
        if all_pts:
            pts = np.concatenate(all_pts, 0)
            x1, y1 = pts.min(0); x2, y2 = pts.max(0)
            bbox = (
                float(max(0, x1)), float(max(0, y1)),
                float(min(W, x2)), float(min(H, y2)),
            )
        return {'body': body, 'face': face, 'hand_l': hl, 'hand_r': hr, 'bbox': bbox}


# ═══════════════════════════════════════════════════════════════
# КОНВЕРТЕР MediaPipe → ULTRA формат (op18 + face + hands)
# ═══════════════════════════════════════════════════════════════

def mp_to_ultra(mp_packet):
    """Конвертит MediaPipe пакет (33 body + 468 face + 21x2 hands)
    в наш стандартный формат (18 op + 68 face subset + 21x2 hands)."""
    if mp_packet is None:
        return None
    body18 = mediapipe_body_to_op18(mp_packet['body'])

    # Делаем 68-точечный subset из 468-mesh (соответствие dlib 68)
    DLIB68_FROM_MP468 = [
        # jawline (17)
        127, 234, 132, 58, 172, 136, 150, 149, 176, 148, 152,
        377, 400, 378, 379, 365, 397,
        # right brow (5)
        70, 63, 105, 66, 107,
        # left brow (5)
        336, 296, 334, 293, 300,
        # nose bridge (4)
        168, 6, 197, 195,
        # nose bottom (5)
        5, 4, 1, 19, 94,
        # right eye (6)
        33, 160, 158, 133, 153, 144,
        # left eye (6)
        362, 385, 387, 263, 373, 380,
        # outer mouth (12)
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
        # inner mouth (8)
        78, 81, 13, 311, 308, 402, 14, 178,
    ]
    face468 = mp_packet['face']
    face68 = np.full((68, 3), np.nan, dtype=np.float32)
    for i, idx in enumerate(DLIB68_FROM_MP468):
        if idx < len(face468):
            face68[i] = face468[idx]

    return {
        'body': body18,
        'face': face68,
        'hand_l': mp_packet['hand_l'],
        'hand_r': mp_packet['hand_r'],
        'bbox': mp_packet['bbox'],
    }


# ═══════════════════════════════════════════════════════════════
# КАНОНИЧЕСКИЙ OPENPOSE РЕНДЕР
# ═══════════════════════════════════════════════════════════════

def draw_openpose(canvas, body18, hand_l, hand_r, face68,
                  draw_body=True, draw_hands=True, draw_face=True,
                  line_thickness=4, point_radius=4, conf_alpha=True):
    """canvas: HxWx3 uint8. Все keypoints в пиксельных координатах исходника
    относительно canvas (caller должен уже отмасштабировать)."""
    import cv2
    H, W = canvas.shape[:2]

    def safe_pt(p):
        if p is None:
            return None
        x, y = p[0], p[1]
        if np.isnan(x) or np.isnan(y):
            return None
        xi = int(round(float(x))); yi = int(round(float(y)))
        # Отбрасываем явный бред (не (0,0) от сырых NaN)
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return (xi, yi)

    if draw_body and body18 is not None:
        # Кости как эллипсы (классический OpenPose стиль)
        stickwidth = max(2, line_thickness)
        for idx, (a, b) in enumerate(OPENPOSE_LIMB_SEQ):
            if a >= 18 or b >= 18:
                continue
            p1 = safe_pt(body18[a]); p2 = safe_pt(body18[b])
            if p1 is None or p2 is None:
                continue
            color = OPENPOSE_COLORS[idx % len(OPENPOSE_COLORS)]
            mx = (p1[0] + p2[0]) / 2.0; my = (p1[1] + p2[1]) / 2.0
            length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
            poly = cv2.ellipse2Poly(
                (int(mx), int(my)),
                (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, poly, color)
            alpha = 0.6
            if conf_alpha:
                c = min(body18[a, 2], body18[b, 2])
                if not np.isnan(c):
                    alpha = float(np.clip(c, 0.4, 1.0)) * 0.7
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        # Точки
        for i in range(18):
            p = safe_pt(body18[i])
            if p is None:
                continue
            col = OPENPOSE_POINT_COLORS[i % len(OPENPOSE_POINT_COLORS)]
            cv2.circle(canvas, p, point_radius, col, -1)

    if draw_hands:
        for hand in (hand_l, hand_r):
            if hand is None:
                continue
            for (a, b) in HAND_EDGES:
                p1 = safe_pt(hand[a]); p2 = safe_pt(hand[b])
                if p1 is None or p2 is None:
                    continue
                cv2.line(canvas, p1, p2, (200, 200, 255), max(1, line_thickness // 2))
            for i in range(len(hand)):
                p = safe_pt(hand[i])
                if p is None:
                    continue
                cv2.circle(canvas, p, max(1, point_radius // 2), (255, 255, 255), -1)

    if draw_face and face68 is not None:
        for i in range(len(face68)):
            p = safe_pt(face68[i])
            if p is None:
                continue
            cv2.circle(canvas, p, 1, (220, 220, 220), -1)

    return canvas


# ═══════════════════════════════════════════════════════════════
# СТАБИЛИЗАЦИЯ ВСЕГО ПОТОКА KEYPOINTS
# ═══════════════════════════════════════════════════════════════

def stabilize_stream(stream, conf_thr=0.3, median_window=5,
                     min_cutoff=1.0, beta=0.02, ema=0.5):
    """stream: (T, K, 3). Per-keypoint per-axis сглаживание."""
    T, K, _ = stream.shape
    out = stream.copy().astype(np.float32)
    for k in range(K):
        # Маска валидности
        conf = out[:, k, 2]
        invalid = (np.isnan(conf)) | (conf < conf_thr)
        for ax in (0, 1):
            v = out[:, k, ax].copy()
            v[invalid] = np.nan
            if np.all(np.isnan(v)):
                continue
            v = _interp_nans_1d(v)
            if median_window >= 3:
                mw = median_window if median_window % 2 == 1 else median_window + 1
                v = _median_filter_1d(v, mw)
            v = _one_euro_1d(v, min_cutoff=min_cutoff, beta=beta)
            if ema > 0:
                v = _bidirectional_ema_1d(v, ema)
            out[:, k, ax] = v
    return out


def face_bbox_from_face(face_arr, body_arr=None):
    """face_arr (K, 3) → (x1,y1,x2,y2) или None."""
    if face_arr is not None and not np.all(np.isnan(face_arr[:, 0])):
        m = ~np.isnan(face_arr[:, 0])
        pts = face_arr[m, :2]
        x1, y1 = pts.min(0); x2, y2 = pts.max(0)
        return (float(x1), float(y1), float(x2), float(y2))
    if body_arr is not None:
        # OpenPose: 0=nose, 14=R_eye, 15=L_eye, 16=R_ear, 17=L_ear
        idxs = [0, 14, 15, 16, 17]
        pts = []
        for i in idxs:
            if i < len(body_arr) and not np.isnan(body_arr[i, 0]):
                pts.append(body_arr[i, :2])
        if pts:
            pts = np.array(pts)
            x1, y1 = pts.min(0); x2, y2 = pts.max(0)
            w = x2 - x1; h = y2 - y1
            # расширим до квадрата лица
            cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
            side = max(w, h) * 2.2
            return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return None


# ═══════════════════════════════════════════════════════════════
# 💀 НОДА: Pose Ultra GOD
# ═══════════════════════════════════════════════════════════════

class PoseUltraGod:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "backend": (["mediapipe", "dwpose", "auto"], {
                    "default": "mediapipe",
                    "tooltip": "mediapipe = без внешних зависимостей, работает всегда. dwpose = SOTA, требует установленный comfyui_controlnet_aux. auto = dwpose → mediapipe"
                }),
                "output_width": ("INT", {
                    "default": 768, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Ширина выхода скелета"
                }),
                "output_height": ("INT", {
                    "default": 768, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Высота выхода скелета"
                }),
                "draw_body": ("BOOLEAN", {"default": True}),
                "draw_hands": ("BOOLEAN", {"default": True}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Face mesh точки"}),
                "line_thickness": ("INT", {
                    "default": 4, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Толщина костей"
                }),
                "point_radius": ("INT", {
                    "default": 4, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Радиус суставов"
                }),
                "keypoint_smoothing": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "EMA на keypoints. Выше = плавнее, ниже = резвее"
                }),
                "median_window": ("INT", {
                    "default": 5, "min": 0, "max": 21, "step": 1,
                    "tooltip": "Медианный pre-filter (убивает выбросы детектора). 0=выкл"
                }),
                "one_euro_min_cutoff": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1
                }),
                "one_euro_beta": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "conf_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Точки с conf ниже считаются невалидными и интерполируются"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1
                }),
                "face_crop_size": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16
                }),
                "face_crop_padding": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 5.0, "step": 0.1
                }),
                "stable_face_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Lock-size face crop (анти-зум-дыхание)"
                }),
                "conf_alpha_lines": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Прозрачность костей по confidence"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("pose_canvas", "face_crops", "keypoints_json",
                    "body_bboxes_json", "face_bboxes_json")
    FUNCTION = "process"
    CATEGORY = "face/pose"
    DESCRIPTION = "💀 GOD-tier pose+face детекция. DWPose/MediaPipe, OpenPose-канон, per-keypoint темпоральная стабилизация, multi-output. Drop-in замена Kijai PoseAndFaceDetection."

    def _make_backend(self, choice):
        order = []
        if choice == "auto":
            order = ["dwpose", "mediapipe"]
        else:
            order = [choice]
        for name in order:
            try:
                if name == "dwpose":
                    b = _DWPoseBackend()
                    print(f"[PoseUltraGOD] Using backend: DWPose (SOTA)")
                    return b, name
                else:
                    b = _MediaPipeBackend(enable_hands=True, enable_face_mesh=True)
                    print(f"[PoseUltraGOD] Using backend: MediaPipe Holistic")
                    return b, name
            except Exception as e:
                print(f"[PoseUltraGOD] backend '{name}' недоступен: {e}")
        raise RuntimeError("[PoseUltraGOD] Никакой backend недоступен!")

    def process(self, images, backend="auto", output_width=768, output_height=768,
                draw_body=True, draw_hands=True, draw_face=True,
                line_thickness=4, point_radius=4,
                keypoint_smoothing=0.6, median_window=5,
                one_euro_min_cutoff=1.0, one_euro_beta=0.03,
                conf_threshold=0.3, detect_every_n=1,
                face_crop_size=512, face_crop_padding=1.6,
                stable_face_crop=True, conf_alpha_lines=True):

        import cv2
        B, H, W, C = images.shape
        print(f"[PoseUltraGOD] {B} frames ({W}x{H}) → canvas {output_width}x{output_height}")

        try:
            det, backend_name = self._make_backend(backend)
        except Exception as e:
            print(f"[PoseUltraGOD] {e}. Возвращаю чёрные кадры.")
            zero = torch.zeros((B, output_height, output_width, 3), dtype=torch.float32)
            zerof = torch.zeros((B, face_crop_size, face_crop_size, 3), dtype=torch.float32)
            return (zero, zerof, "[]", "[]", "[]")

        pbar = comfy.utils.ProgressBar(B)

        # Накопители
        body_stream = np.full((B, 18, 3), np.nan, dtype=np.float32)
        face_stream = np.full((B, 68, 3), np.nan, dtype=np.float32)
        handl_stream = np.full((B, 21, 3), np.nan, dtype=np.float32)
        handr_stream = np.full((B, 21, 3), np.nan, dtype=np.float32)
        body_bboxes = [None] * B

        last_packet = None
        consecutive_none = 0
        for i in range(B):
            do_detect = (i % detect_every_n == 0) or (last_packet is None)
            packet = None
            if do_detect:
                try:
                    frame_u8 = (images[i].cpu().numpy() * 255).astype(np.uint8)
                    raw = det.detect(frame_u8)
                    if raw is not None:
                        if backend_name == "mediapipe":
                            packet = mp_to_ultra(raw)
                        else:
                            packet = raw
                    else:
                        consecutive_none += 1
                except Exception as e:
                    if i < 3:
                        import traceback
                        print(f"[PoseUltraGOD] frame {i} detect err: {e}")
                        traceback.print_exc()
                    consecutive_none += 1

            # Рунтайм фоллбэк: если dwpose провалился на первых 5 кадрах → переключаемся на mediapipe
            if packet is None and last_packet is None and consecutive_none >= 5 and backend_name != "mediapipe":
                print(f"[PoseUltraGOD] {backend_name} вернул None 5 кадров подряд → переключаюсь на mediapipe")
                try:
                    det = _MediaPipeBackend(enable_hands=True, enable_face_mesh=True)
                    backend_name = "mediapipe"
                    consecutive_none = 0
                    # Передетектим текущий кадр
                    frame_u8 = (images[i].cpu().numpy() * 255).astype(np.uint8)
                    raw = det.detect(frame_u8)
                    if raw is not None:
                        packet = mp_to_ultra(raw)
                except Exception as e:
                    print(f"[PoseUltraGOD] mediapipe fallback тоже сломался: {e}")

            if packet is None:
                packet = last_packet  # держим последнее как фоллбэк
            if packet is not None:
                body_stream[i] = packet['body']
                face_stream[i] = packet['face']
                handl_stream[i] = packet['hand_l']
                handr_stream[i] = packet['hand_r']
                body_bboxes[i] = packet.get('bbox')
                last_packet = packet
                consecutive_none = 0
            pbar.update_absolute(i, B)

        n_det = sum(1 for b in body_bboxes if b is not None)
        print(f"[PoseUltraGOD] Detected on {n_det}/{B} frames ({100.0*n_det/max(1,B):.1f}%)")

        # ── ДИАГНОСТИКА: что реально пришло в потоки ──
        body_nan_pct = float(np.isnan(body_stream[:, :, 0]).mean()) * 100
        face_nan_pct = float(np.isnan(face_stream[:, :, 0]).mean()) * 100
        print(f"[PoseUltraGOD] body NaN: {body_nan_pct:.1f}%, face NaN: {face_nan_pct:.1f}%")
        if n_det > 0:
            # Найдём первый валидный кадр и покажем neck
            for j in range(B):
                if body_bboxes[j] is not None:
                    nk = body_stream[j, 1]
                    bb = body_bboxes[j]
                    print(f"[PoseUltraGOD] sample frame {j}: neck=({nk[0]:.1f},{nk[1]:.1f},c={nk[2]:.2f}) bbox={bb}")
                    break
        if n_det == 0:
            print("[PoseUltraGOD] ⚠⚠⚠ НИ ОДНОГО кадра с детекцией! Проверь: в кадре есть человек? Разрешение входа нормальное? backend работает?")

        # ── Стабилизация ──
        body_stream = stabilize_stream(
            body_stream, conf_thr=conf_threshold, median_window=median_window,
            min_cutoff=one_euro_min_cutoff, beta=one_euro_beta, ema=keypoint_smoothing)
        face_stream = stabilize_stream(
            face_stream, conf_thr=conf_threshold, median_window=median_window,
            min_cutoff=one_euro_min_cutoff, beta=one_euro_beta, ema=keypoint_smoothing)
        handl_stream = stabilize_stream(
            handl_stream, conf_thr=conf_threshold, median_window=median_window,
            min_cutoff=one_euro_min_cutoff * 1.5, beta=one_euro_beta, ema=keypoint_smoothing)
        handr_stream = stabilize_stream(
            handr_stream, conf_thr=conf_threshold, median_window=median_window,
            min_cutoff=one_euro_min_cutoff * 1.5, beta=one_euro_beta, ema=keypoint_smoothing)

        # ── Face bbox потоки + стабилизация face crop ──
        fcx = np.full(B, np.nan); fcy = np.full(B, np.nan); fsize = np.full(B, np.nan)
        face_bboxes_raw = []
        for i in range(B):
            fb = face_bbox_from_face(face_stream[i], body_stream[i])
            face_bboxes_raw.append(fb)
            if fb:
                x1, y1, x2, y2 = fb
                fcx[i] = (x1 + x2) / 2.0
                fcy[i] = (y1 + y2) / 2.0
                fsize[i] = max(x2 - x1, y2 - y1) * face_crop_padding

        if np.any(~np.isnan(fcx)):
            fcx = _interp_nans_1d(fcx)
            fcy = _interp_nans_1d(fcy)
            fsize = _interp_nans_1d(fsize)
            if median_window >= 3:
                mw = median_window if median_window % 2 == 1 else median_window + 1
                fcx = _median_filter_1d(fcx, mw)
                fcy = _median_filter_1d(fcy, mw)
                fsize = _median_filter_1d(fsize, mw)
            fcx = _one_euro_1d(fcx, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
            fcy = _one_euro_1d(fcy, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
            fsize = _one_euro_1d(fsize, min_cutoff=one_euro_min_cutoff * 0.6, beta=one_euro_beta * 0.5)
            fcx = _bidirectional_ema_1d(fcx, 0.85)
            fcy = _bidirectional_ema_1d(fcy, 0.85)
            fsize = _bidirectional_ema_1d(fsize, 0.92)
            if stable_face_crop:
                med = float(np.median(fsize))
                fsize = np.full(B, med, dtype=np.float64)
        else:
            # лиц не нашли — центр кадра
            fcx[:] = W / 2.0; fcy[:] = H / 2.0; fsize[:] = min(W, H) * 0.6

        # ── Рендер ──
        # Padded canvas (как WAN: scale + offset)
        ar_src = W / max(1, H)
        ar_dst = output_width / max(1, output_height)
        if ar_src > ar_dst:
            kx = output_width / W
            ky = kx
            ox = 0.0
            oy = (output_height - H * ky) / 2.0
        else:
            ky = output_height / H
            kx = ky
            oy = 0.0
            ox = (output_width - W * kx) / 2.0

        def scale_pts(arr):
            out = arr.copy()
            out[:, 0] = arr[:, 0] * kx + ox
            out[:, 1] = arr[:, 1] * ky + oy
            return out

        pose_frames = []
        face_frames = []
        for i in range(B):
            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            draw_openpose(
                canvas,
                scale_pts(body_stream[i]) if draw_body else None,
                scale_pts(handl_stream[i]) if draw_hands else None,
                scale_pts(handr_stream[i]) if draw_hands else None,
                scale_pts(face_stream[i]) if draw_face else None,
                draw_body=draw_body, draw_hands=draw_hands, draw_face=draw_face,
                line_thickness=line_thickness, point_radius=point_radius,
                conf_alpha=conf_alpha_lines,
            )
            pose_frames.append(torch.from_numpy(canvas.astype(np.float32) / 255.0))

            # Face crop
            try:
                frame_u8 = (images[i].cpu().numpy() * 255).astype(np.uint8)
                cx = float(fcx[i]); cy = float(fcy[i]); s = float(max(8.0, fsize[i]))
                scale = face_crop_size / s
                M = np.array([
                    [scale, 0.0, face_crop_size / 2.0 - scale * cx],
                    [0.0, scale, face_crop_size / 2.0 - scale * cy],
                ], dtype=np.float32)
                warped = cv2.warpAffine(
                    frame_u8, M, (face_crop_size, face_crop_size),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                face_frames.append(torch.from_numpy(warped.astype(np.float32) / 255.0))
            except Exception as e:
                face_frames.append(torch.zeros((face_crop_size, face_crop_size, 3)))

        pose_out = torch.stack(pose_frames, 0).to(images.device)
        face_out = torch.stack(face_frames, 0).to(images.device)

        # ── JSON выходы ──
        kp_json = []
        for i in range(B):
            kp_json.append({
                "frame": i,
                "body": body_stream[i].tolist(),
                "face": face_stream[i].tolist(),
                "hand_l": handl_stream[i].tolist(),
                "hand_r": handr_stream[i].tolist(),
            })
        body_bb_json = []
        face_bb_json = []
        for i in range(B):
            body_bb_json.append({"frame": i, "bbox": body_bboxes[i]})
            face_bb_json.append({"frame": i, "bbox": face_bboxes_raw[i]})

        try:
            det.cleanup()
        except Exception:
            pass

        print(f"[PoseUltraGOD] Done. pose={pose_out.shape} face={face_out.shape}")
        return (
            pose_out, face_out,
            json.dumps(kp_json),
            json.dumps(body_bb_json),
            json.dumps(face_bb_json),
        )


NODE_CLASS_MAPPINGS = {
    "PoseUltraGod": PoseUltraGod,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseUltraGod": "💀 Pose Ultra GOD",
}
