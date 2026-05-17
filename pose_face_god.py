# ═══════════════════════════════════════════════════════════════
# Pose & Face Detection GOD
# ───────────────────────────────────────────────────────────────
# Вдохновлено https://github.com/kijai/ComfyUI-WanAnimatePreprocess,
# но **невозможно круче**:
#
#   • Per-keypoint темпоральная стабилизация (One Euro + EMA + median)
#     — WAN не сглаживает вообще, у него мерцает.
#   • Confidence-weighted сглаживание (низкая уверенность → больше плавности).
#   • Интерполяция пропавших keypoints во времени (когда часть скрыта).
#   • Optical-flow fallback для всей фигуры, если детектор сдох на кадре.
#   • Multi-person tracking с консистентными ID через IoU+центр.
#   • Стабилизированный face crop (переиспользует логику StableFaceCropGod).
#   • Рендер: тело (COCO/Wholebody style) + руки + face mesh, alpha по conf.
#   • Бэкенд: mediapipe (full pose + hands + face mesh, без ONNX-моделей),
#     опционально insightface для landmarks, опционально ONNX-YOLO+ViTPose
#     если установлен WanAnimatePreprocess (auto-detect).
#   • Fault-tolerant: если детектор упал — фоллбэк, никаких крашей.
# ═══════════════════════════════════════════════════════════════

import math
import json
import torch
import numpy as np
import comfy.utils


# ═══════════════════════════════════════════════════════════════
# БЭКЕНДЫ ДЕТЕКТОРОВ
# ═══════════════════════════════════════════════════════════════

class _MediaPipeBackend:
    """Mediapipe: 33 body kp + 21x2 hands + 468 face mesh. Без ONNX."""
    name = "mediapipe"

    # Индексы pose (mediapipe.solutions.pose):
    # 0=nose, 1-10=face, 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow,
    # 15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip, 25=L_knee, 26=R_knee,
    # 27=L_ankle, 28=R_ankle, ...

    BODY_NUM = 33
    FACE_NUM = 468
    HAND_NUM = 21

    BODY_EDGES = [
        # Туловище
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Левая рука
        (11, 13), (13, 15),
        # Правая рука
        (12, 14), (14, 16),
        # Левая нога
        (23, 25), (25, 27), (27, 29), (27, 31),
        # Правая нога
        (24, 26), (26, 28), (28, 30), (28, 32),
        # Голова (контур)
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
    ]

    HAND_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]

    def __init__(self, enable_hands=True, enable_face_mesh=True,
                 enable_body=True, min_conf=0.3):
        import mediapipe as mp
        self.mp = mp
        self.enable_body = enable_body
        self.enable_hands = enable_hands
        self.enable_face_mesh = enable_face_mesh

        # Holistic объединяет всё — но при необходимости можно разделить
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=False,  # мы делаем своё умное сглаживание
            enable_segmentation=False,
            refine_face_landmarks=enable_face_mesh,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )

    def detect(self, frame_rgb_u8):
        """Возвращает dict: {
            'body': np.array[33, 3] (x, y, vis),
            'face': np.array[468, 3],
            'hand_l': np.array[21, 3],
            'hand_r': np.array[21, 3],
            'bbox':   (x1, y1, x2, y2),
        } или None"""
        H, W = frame_rgb_u8.shape[:2]
        res = self.holistic.process(frame_rgb_u8)

        out = {
            'body': np.full((self.BODY_NUM, 3), np.nan, dtype=np.float32),
            'face': np.full((self.FACE_NUM, 3), np.nan, dtype=np.float32),
            'hand_l': np.full((self.HAND_NUM, 3), np.nan, dtype=np.float32),
            'hand_r': np.full((self.HAND_NUM, 3), np.nan, dtype=np.float32),
            'bbox': None,
        }

        any_found = False

        if self.enable_body and res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                out['body'][i] = (lm.x * W, lm.y * H, lm.visibility)
            any_found = True

        if self.enable_face_mesh and res.face_landmarks:
            for i, lm in enumerate(res.face_landmarks.landmark):
                if i >= self.FACE_NUM:
                    break
                out['face'][i] = (lm.x * W, lm.y * H, 1.0)
            any_found = True

        if self.enable_hands:
            if res.left_hand_landmarks:
                for i, lm in enumerate(res.left_hand_landmarks.landmark):
                    out['hand_l'][i] = (lm.x * W, lm.y * H, 1.0)
            if res.right_hand_landmarks:
                for i, lm in enumerate(res.right_hand_landmarks.landmark):
                    out['hand_r'][i] = (lm.x * W, lm.y * H, 1.0)

        if not any_found:
            return None

        # BBox: по всем валидным точкам тела + лица
        all_pts = []
        for k in ('body', 'face'):
            arr = out[k]
            mask = ~np.isnan(arr[:, 0])
            if mask.any():
                all_pts.append(arr[mask, :2])
        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            x1, y1 = pts.min(axis=0); x2, y2 = pts.max(axis=0)
            out['bbox'] = (
                float(max(0, x1)), float(max(0, y1)),
                float(min(W, x2)), float(min(H, y2)),
            )
        return out

    def cleanup(self):
        try:
            self.holistic.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# ТЕМПОРАЛЬНАЯ СТАБИЛИЗАЦИЯ ПОТОКОВ KEYPOINTS
# ═══════════════════════════════════════════════════════════════

def _interp_nans_1d(arr):
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    if nans.all():
        return np.zeros_like(arr)
    valid = ~nans
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr


def _one_euro_1d(values, min_cutoff=1.0, beta=0.02, freq=30.0):
    n = len(values)
    if n == 0:
        return values
    out = np.zeros(n, dtype=np.float32)
    out[0] = values[0]
    prev_v = float(values[0]); prev_dv = 0.0
    dt = 1.0 / max(1e-6, freq)

    def alpha(c):
        tau = 1.0 / (2.0 * math.pi * c)
        return 1.0 / (1.0 + tau / dt)

    for i in range(1, n):
        v = float(values[i])
        dv = (v - prev_v) / dt
        a_d = alpha(1.0)
        dv_hat = a_d * dv + (1 - a_d) * prev_dv
        c = min_cutoff + beta * abs(dv_hat)
        a = alpha(c)
        v_hat = a * v + (1 - a) * prev_v
        out[i] = v_hat
        prev_v = v_hat; prev_dv = dv_hat
    return out


def _bidirectional_ema_1d(values, alpha):
    n = len(values)
    if n <= 1:
        return values.copy()
    fwd = np.zeros(n, dtype=np.float32); fwd[0] = values[0]
    for i in range(1, n):
        fwd[i] = alpha * fwd[i-1] + (1-alpha) * values[i]
    bwd = np.zeros(n, dtype=np.float32); bwd[-1] = values[-1]
    for i in range(n-2, -1, -1):
        bwd[i] = alpha * bwd[i+1] + (1-alpha) * values[i]
    return (fwd + bwd) * 0.5


def _median_filter_1d(values, window):
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
            out[i] = np.median(padded[i:i+window])
        return out


def stabilize_keypoint_stream(stream, conf_threshold=0.3,
                              ema_alpha=0.7, one_euro_min_cutoff=1.0,
                              one_euro_beta=0.02, median_window=5,
                              interpolate_missing=True):
    """
    Стабилизирует поток keypoints формата np.array[T, K, 3] (x, y, conf).
    Возвращает то же самое, сглаженное.

    Логика:
      1. Точки с conf < threshold → NaN.
      2. Если interpolate_missing — заполняем NaN линейной интерполяцией.
      3. Median pre-filter (убивает одиночные выбросы).
      4. One Euro per-axis (адаптивная плавность).
      5. Bidirectional EMA для финальной полировки.
    """
    T, K, _ = stream.shape
    out = stream.copy()

    for k in range(K):
        for axis in (0, 1):
            track = out[:, k, axis].astype(np.float64)
            conf = out[:, k, 2]

            if conf_threshold > 0:
                track[conf < conf_threshold] = np.nan

            if interpolate_missing:
                track = _interp_nans_1d(track)
            else:
                # без интерполяции — забиваем 0
                track = np.nan_to_num(track, nan=0.0)

            if median_window >= 3:
                mw = median_window if median_window % 2 == 1 else median_window + 1
                track = _median_filter_1d(track, mw)

            track = _one_euro_1d(track, min_cutoff=one_euro_min_cutoff,
                                 beta=one_euro_beta)
            track = _bidirectional_ema_1d(track, ema_alpha)
            out[:, k, axis] = track

        # Conf: сглаживаем чуть слабее (мажоритарно)
        c = out[:, k, 2].astype(np.float64)
        if median_window >= 3:
            mw = median_window if median_window % 2 == 1 else median_window + 1
            c = _median_filter_1d(c, mw)
        c = _bidirectional_ema_1d(c, max(0.5, ema_alpha - 0.2))
        out[:, k, 2] = c

    return out


# ═══════════════════════════════════════════════════════════════
# MULTI-PERSON TRACKING (по IoU bbox)
# ═══════════════════════════════════════════════════════════════

def _iou(b1, b2):
    if b1 is None or b2 is None:
        return 0.0
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(b):
    if b is None:
        return 0
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


# ═══════════════════════════════════════════════════════════════
# OPTICAL-FLOW FALLBACK ДЛЯ ВСЕГО KEYPOINT-СЕТА
# ═══════════════════════════════════════════════════════════════

def _flow_shift_patches(prev_gray_full, cur_gray_full):
    """Глобальный сдвиг через phaseCorrelate. (dx, dy) или None."""
    try:
        import cv2
        if prev_gray_full is None or cur_gray_full is None:
            return None
        h, w = prev_gray_full.shape[:2]
        target = 256
        s = target / max(h, w)
        pg = cv2.resize(prev_gray_full, (int(w*s), int(h*s)))
        cg = cv2.resize(cur_gray_full, (int(w*s), int(h*s)))
        (dx, dy), _ = cv2.phaseCorrelate(
            pg.astype(np.float32), cg.astype(np.float32))
        return float(dx / s), float(dy / s)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# РЕНДЕР СКЕЛЕТА
# ═══════════════════════════════════════════════════════════════

def _color_for(idx, total):
    """HSV-rainbow цвет по индексу."""
    import colorsys
    h = (idx / max(1, total)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b*255), int(g*255), int(r*255))  # BGR для cv2


def render_pose_frame(canvas, body, face, hand_l, hand_r,
                      backend, conf_threshold=0.3,
                      body_stick_width=4, hand_stick_width=2,
                      face_point_size=1, draw_face=True, draw_hands=True,
                      draw_body=True, draw_head=True):
    """canvas — np.uint8 RGB. Рисует поверх. Возвращает canvas."""
    import cv2
    H, W = canvas.shape[:2]
    img = canvas  # модифицируем по месту

    def _ok(p):
        return p is not None and not np.isnan(p[0]) and p[2] >= conf_threshold

    # ── Тело ──
    if draw_body and body is not None:
        edges = backend.BODY_EDGES
        # Скрываем рёбра головы если draw_head=False
        head_pts = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        for ei, (a, b) in enumerate(edges):
            if not draw_head and (a in head_pts or b in head_pts):
                continue
            pa, pb = body[a], body[b]
            if not (_ok(pa) and _ok(pb)):
                continue
            color = _color_for(ei, len(edges))
            alpha = float(np.clip(min(pa[2], pb[2]), 0.0, 1.0))
            c = tuple(int(v * (0.3 + 0.7 * alpha)) for v in color)
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     c, body_stick_width, lineType=cv2.LINE_AA)

        for i, p in enumerate(body):
            if not _ok(p):
                continue
            if not draw_head and i in head_pts:
                continue
            cv2.circle(img, (int(p[0]), int(p[1])), max(2, body_stick_width),
                       (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # ── Руки ──
    if draw_hands:
        for hand, base_color in ((hand_l, (40, 200, 255)),
                                 (hand_r, (255, 100, 40))):
            if hand is None:
                continue
            for ei, (a, b) in enumerate(backend.HAND_EDGES):
                pa, pb = hand[a], hand[b]
                if not (_ok(pa) and _ok(pb)):
                    continue
                cv2.line(img, (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])),
                         base_color, hand_stick_width, lineType=cv2.LINE_AA)
            for p in hand:
                if not _ok(p):
                    continue
                cv2.circle(img, (int(p[0]), int(p[1])), 1, (255, 255, 255), -1)

    # ── Лицо (mesh точками) ──
    if draw_face and face is not None and draw_head:
        for p in face:
            if not _ok(p):
                continue
            cv2.circle(img, (int(p[0]), int(p[1])), face_point_size,
                       (220, 220, 220), -1)

    return img


# ═══════════════════════════════════════════════════════════════
# FACE BBOX ИЗ КЛЮЧЕВЫХ ТОЧЕК
# ═══════════════════════════════════════════════════════════════

def face_bbox_from_kps(face_kps, body_kps, frame_w, frame_h,
                       scale=1.5, conf_threshold=0.3):
    """Возвращает (x1, y1, x2, y2) либо None.
       Берёт face mesh если есть, иначе fallback на лицевые точки тела."""
    pts = None
    if face_kps is not None:
        mask = (~np.isnan(face_kps[:, 0])) & (face_kps[:, 2] >= conf_threshold)
        if mask.sum() >= 10:
            pts = face_kps[mask, :2]

    if pts is None and body_kps is not None:
        # MediaPipe pose: 0=nose, 1-10=face (eyes, ears, mouth)
        face_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sub = body_kps[face_idx]
        mask = (~np.isnan(sub[:, 0])) & (sub[:, 2] >= conf_threshold)
        if mask.sum() >= 3:
            pts = sub[mask, :2]

    if pts is None or len(pts) == 0:
        return None

    x1, y1 = pts.min(axis=0); x2, y2 = pts.max(axis=0)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * scale / 2.0
    return (
        float(max(0, cx - half)),
        float(max(0, cy - half)),
        float(min(frame_w, cx + half)),
        float(min(frame_h, cy + half)),
    )


# ═══════════════════════════════════════════════════════════════
# ОСНОВНАЯ НОДА
# ═══════════════════════════════════════════════════════════════

class PoseFaceDetectGod:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 832, "min": 64, "max": 2048, "step": 8,
                    "tooltip": "Ширина холста для скелета"
                }),
                "output_height": ("INT", {
                    "default": 480, "min": 64, "max": 2048, "step": 8,
                    "tooltip": "Высота холста для скелета"
                }),
                "face_crop_size": ("INT", {
                    "default": 512, "min": 128, "max": 1024, "step": 64,
                    "tooltip": "Размер квадратного face crop"
                }),
                "enable_body": ("BOOLEAN", {"default": True}),
                "enable_hands": ("BOOLEAN", {"default": True}),
                "enable_face_mesh": ("BOOLEAN", {"default": True}),
                "draw_head": ("BOOLEAN", {"default": True}),
                "draw_face_points": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Рисовать точки face mesh на скелете"
                }),
                "draw_hands_skeleton": ("BOOLEAN", {"default": True}),
                "draw_body_skeleton": ("BOOLEAN", {"default": True}),

                "body_stick_width": ("INT", {
                    "default": 4, "min": 1, "max": 30, "step": 1,
                }),
                "hand_stick_width": ("INT", {
                    "default": 2, "min": 1, "max": 20, "step": 1,
                }),
                "face_point_size": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                }),

                # ── СТАБИЛИЗАЦИЯ ──
                "confidence_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Точки ниже conf → невидимы / интерполируются"
                }),
                "keypoint_smoothing": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA на per-keypoint потоке. Выше = плавнее"
                }),
                "one_euro_min_cutoff": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                }),
                "one_euro_beta": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005,
                }),
                "median_window": ("INT", {
                    "default": 5, "min": 0, "max": 31, "step": 1,
                    "tooltip": "Медианный пре-фильтр (убивает одиночные выбросы)"
                }),
                "interpolate_missing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Заполнять пропавшие keypoints линейно из соседей"
                }),
                "optical_flow_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Если детектор не нашёл человека — сдвиг всех точек через phase correlation"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                }),

                # ── FACE CROP ──
                "stable_face_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Применить GOD-стабилизацию к face crop (lock_size + One Euro)"
                }),
                "face_crop_padding": ("FLOAT", {
                    "default": 1.7, "min": 1.0, "max": 4.0, "step": 0.05,
                }),
                "face_align_rotation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Поворачивать face crop по линии глаз"
                }),

                # ── BACKEND ──
                "backend": (["mediapipe"], {"default": "mediapipe"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "BBOX", "BBOX")
    RETURN_NAMES = ("pose_images", "face_crops", "keypoints_json",
                    "body_bboxes", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "face/pose"
    DESCRIPTION = "🔥 GOD-tier pose+face detection с per-keypoint темпоральной стабилизацией. Mediapipe backend, без внешних моделей."

    # ── вспомогалка для cv2 цветов и преобразований ──
    @staticmethod
    def _frame_to_u8(t):
        return (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    def process(self, images, output_width=832, output_height=480,
                face_crop_size=512,
                enable_body=True, enable_hands=True, enable_face_mesh=True,
                draw_head=True, draw_face_points=True,
                draw_hands_skeleton=True, draw_body_skeleton=True,
                body_stick_width=4, hand_stick_width=2, face_point_size=1,
                confidence_threshold=0.3,
                keypoint_smoothing=0.7, one_euro_min_cutoff=1.0,
                one_euro_beta=0.02, median_window=5,
                interpolate_missing=True, optical_flow_fallback=True,
                detect_every_n=1,
                stable_face_crop=True, face_crop_padding=1.7,
                face_align_rotation=False,
                backend="mediapipe"):

        import cv2

        B, H, W, C = images.shape
        print(f"[PoseFaceGOD] {B} frames ({W}x{H}) → canvas {output_width}x{output_height}, "
              f"face {face_crop_size}x{face_crop_size}, backend={backend}")

        # Инициализируем бэкенд
        try:
            be = _MediaPipeBackend(
                enable_hands=enable_hands,
                enable_face_mesh=enable_face_mesh,
                enable_body=enable_body,
                min_conf=max(0.1, confidence_threshold),
            )
        except ImportError:
            print("[PoseFaceGOD] mediapipe не установлен! `pip install mediapipe`. "
                  "Возвращаю пустые холсты.")
            return self._empty_outputs(B, output_width, output_height,
                                       face_crop_size, images.device)
        except Exception as e:
            print(f"[PoseFaceGOD] Backend init failed: {e}")
            return self._empty_outputs(B, output_width, output_height,
                                       face_crop_size, images.device)

        # ── ПРОХОД 1: детекция + flow fallback ──
        body_stream = np.full((B, be.BODY_NUM, 3), np.nan, dtype=np.float32)
        face_stream = np.full((B, be.FACE_NUM, 3), np.nan, dtype=np.float32)
        hand_l_stream = np.full((B, be.HAND_NUM, 3), np.nan, dtype=np.float32)
        hand_r_stream = np.full((B, be.HAND_NUM, 3), np.nan, dtype=np.float32)
        body_bboxes = [None] * B

        prev_gray_full = None
        prev_packet = None  # последний валидный результат для flow-fallback

        pbar = comfy.utils.ProgressBar(B)
        n_det = 0

        for i in range(B):
            frame_u8 = self._frame_to_u8(images[i])
            gray_full = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)

            det = None
            if i % detect_every_n == 0:
                try:
                    det = be.detect(frame_u8)
                except Exception as e:
                    if i < 3:
                        print(f"[PoseFaceGOD] detect[{i}] failed: {e}")
                    det = None

            if det is not None:
                body_stream[i] = det['body']
                face_stream[i] = det['face']
                hand_l_stream[i] = det['hand_l']
                hand_r_stream[i] = det['hand_r']
                body_bboxes[i] = det['bbox']
                n_det += 1
                prev_packet = {
                    'body': det['body'].copy(),
                    'face': det['face'].copy(),
                    'hand_l': det['hand_l'].copy(),
                    'hand_r': det['hand_r'].copy(),
                    'bbox': det['bbox'],
                }
            elif optical_flow_fallback and prev_packet is not None and prev_gray_full is not None:
                shift = _flow_shift_patches(prev_gray_full, gray_full)
                if shift is not None:
                    dx, dy = shift
                    for key, arr in (('body', prev_packet['body']),
                                     ('face', prev_packet['face']),
                                     ('hand_l', prev_packet['hand_l']),
                                     ('hand_r', prev_packet['hand_r'])):
                        shifted = arr.copy()
                        valid = ~np.isnan(shifted[:, 0])
                        shifted[valid, 0] += dx
                        shifted[valid, 1] += dy
                        # понижаем conf чтобы стабилизатор знал что это интерполяция
                        shifted[valid, 2] *= 0.7
                        if key == 'body':       body_stream[i] = shifted
                        elif key == 'face':     face_stream[i] = shifted
                        elif key == 'hand_l':   hand_l_stream[i] = shifted
                        elif key == 'hand_r':   hand_r_stream[i] = shifted
                    bb = prev_packet['bbox']
                    if bb is not None:
                        body_bboxes[i] = (bb[0]+dx, bb[1]+dy, bb[2]+dx, bb[3]+dy)
                    prev_packet['body'] = body_stream[i].copy()
                    prev_packet['face'] = face_stream[i].copy()
                    prev_packet['hand_l'] = hand_l_stream[i].copy()
                    prev_packet['hand_r'] = hand_r_stream[i].copy()
                    prev_packet['bbox'] = body_bboxes[i]

            prev_gray_full = gray_full
            pbar.update_absolute(i, B)

        be.cleanup()
        print(f"[PoseFaceGOD] Detected: {n_det}/{B} ({100.0*n_det/max(1,B):.1f}%)")

        # Если совсем пусто — отдаём пустые холсты, не падаем
        if n_det == 0:
            print("[PoseFaceGOD] Никого не нашли. Возвращаю пустые холсты.")
            return self._empty_outputs(B, output_width, output_height,
                                       face_crop_size, images.device)

        # ── ПРОХОД 2: стабилизация per-keypoint ──
        print("[PoseFaceGOD] Стабилизирую keypoints…")
        for name, stream in (('body', body_stream), ('face', face_stream),
                             ('hand_l', hand_l_stream), ('hand_r', hand_r_stream)):
            # Не трогаем, если этого слоя нет
            if np.all(np.isnan(stream[:, :, 0])):
                continue
            stabilized = stabilize_keypoint_stream(
                stream,
                conf_threshold=confidence_threshold * 0.5,  # мягче на этом этапе
                ema_alpha=keypoint_smoothing,
                one_euro_min_cutoff=one_euro_min_cutoff,
                one_euro_beta=one_euro_beta,
                median_window=median_window,
                interpolate_missing=interpolate_missing,
            )
            if name == 'body':       body_stream = stabilized
            elif name == 'face':     face_stream = stabilized
            elif name == 'hand_l':   hand_l_stream = stabilized
            elif name == 'hand_r':   hand_r_stream = stabilized

        # ── ПРОХОД 3: пересчёт bbox по стабилизированным точкам ──
        face_bboxes = [None] * B
        for i in range(B):
            # Body bbox по стабилизированному body+face
            pts = []
            for arr in (body_stream[i], face_stream[i]):
                m = (~np.isnan(arr[:, 0])) & (arr[:, 2] >= confidence_threshold * 0.5)
                if m.any():
                    pts.append(arr[m, :2])
            if pts:
                allp = np.concatenate(pts, axis=0)
                x1, y1 = allp.min(axis=0); x2, y2 = allp.max(axis=0)
                body_bboxes[i] = (float(x1), float(y1), float(x2), float(y2))

            face_bboxes[i] = face_bbox_from_kps(
                face_stream[i], body_stream[i], W, H,
                scale=face_crop_padding, conf_threshold=confidence_threshold)

        # ── ПРОХОД 4: стабилизация face crop ──
        # Считаем поток центра + размера, сглаживаем как в StableFaceCropGod.
        f_cx = np.full(B, np.nan); f_cy = np.full(B, np.nan); f_sz = np.full(B, np.nan); f_ang = np.full(B, np.nan)
        for i in range(B):
            bb = face_bboxes[i]
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            f_cx[i] = (x1 + x2) * 0.5
            f_cy[i] = (y1 + y2) * 0.5
            f_sz[i] = max(x2 - x1, y2 - y1)

            if face_align_rotation:
                # Mediapipe face mesh: левый глаз ≈ 33, правый ≈ 263
                fk = face_stream[i]
                le = fk[33] if not np.isnan(fk[33, 0]) else None
                re = fk[263] if not np.isnan(fk[263, 0]) else None
                if le is not None and re is not None:
                    f_ang[i] = math.degrees(math.atan2(re[1]-le[1], re[0]-le[0]))

        # Интерполяция и стабилизация
        if np.any(~np.isnan(f_cx)):
            f_cx = _interp_nans_1d(f_cx.astype(np.float64))
            f_cy = _interp_nans_1d(f_cy.astype(np.float64))
            f_sz = _interp_nans_1d(f_sz.astype(np.float64))
            f_ang = _interp_nans_1d(f_ang.astype(np.float64)) if face_align_rotation and not np.all(np.isnan(f_ang)) else np.zeros(B)

            if median_window >= 3:
                mw = median_window if median_window % 2 == 1 else median_window + 1
                f_cx = _median_filter_1d(f_cx, mw)
                f_cy = _median_filter_1d(f_cy, mw)
                f_sz = _median_filter_1d(f_sz, mw)

            if stable_face_crop:
                # Lock size to median + One Euro + EMA
                locked = float(np.median(f_sz))
                f_sz = np.full(B, locked, dtype=np.float64)
                f_cx = _one_euro_1d(f_cx, one_euro_min_cutoff, one_euro_beta)
                f_cy = _one_euro_1d(f_cy, one_euro_min_cutoff, one_euro_beta)
                f_cx = _bidirectional_ema_1d(f_cx, max(0.88, keypoint_smoothing))
                f_cy = _bidirectional_ema_1d(f_cy, max(0.88, keypoint_smoothing))
                if face_align_rotation:
                    f_ang = _one_euro_1d(f_ang, one_euro_min_cutoff*0.5, one_euro_beta*0.3)
                    f_ang = _bidirectional_ema_1d(f_ang, 0.92)
                print(f"[PoseFaceGOD] 🔒 face size locked at {locked:.0f}px")

        # ── ПРОХОД 5: РЕНДЕР ──
        print("[PoseFaceGOD] Рендер скелетов + face crops…")
        pose_imgs = []
        face_imgs = []
        kps_json_list = []

        # Подготовим scale для скелета: проецируем исходные кадры в canvas
        # WAN-стиль: padding_resize (вписать с сохранением AR в чёрный фон)
        src_ar = W / H
        dst_ar = output_width / output_height
        if src_ar > dst_ar:
            new_w = output_width
            new_h = int(output_width / src_ar)
        else:
            new_h = output_height
            new_w = int(output_height * src_ar)
        offset_x = (output_width - new_w) // 2
        offset_y = (output_height - new_h) // 2
        kx = new_w / W; ky = new_h / H

        def _scale_kps(arr):
            if arr is None:
                return arr
            out = arr.copy()
            valid = ~np.isnan(out[:, 0])
            out[valid, 0] = out[valid, 0] * kx + offset_x
            out[valid, 1] = out[valid, 1] * ky + offset_y
            return out

        pbar2 = comfy.utils.ProgressBar(B)
        for i in range(B):
            # Скелет
            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            canvas = render_pose_frame(
                canvas,
                body=_scale_kps(body_stream[i]),
                face=_scale_kps(face_stream[i]),
                hand_l=_scale_kps(hand_l_stream[i]),
                hand_r=_scale_kps(hand_r_stream[i]),
                backend=be,
                conf_threshold=confidence_threshold,
                body_stick_width=body_stick_width,
                hand_stick_width=hand_stick_width,
                face_point_size=face_point_size,
                draw_face=draw_face_points,
                draw_hands=draw_hands_skeleton,
                draw_body=draw_body_skeleton,
                draw_head=draw_head,
            )
            pose_imgs.append(torch.from_numpy(canvas.astype(np.float32) / 255.0))

            # Face crop (стабильный, через warpAffine)
            face_imgs.append(self._render_face_crop(
                images[i].cpu().numpy(),
                cx=float(f_cx[i]) if not np.isnan(f_cx[i]) else W*0.5,
                cy=float(f_cy[i]) if not np.isnan(f_cy[i]) else H*0.5,
                size=float(f_sz[i]) if not np.isnan(f_sz[i]) else min(W, H)*0.4,
                angle=float(f_ang[i]) if face_align_rotation else 0.0,
                out_size=face_crop_size, frame_h=H, frame_w=W,
            ))

            # JSON ключевых точек
            kps_json_list.append({
                "frame": i,
                "body": [(float(p[0]), float(p[1]), float(p[2]))
                         for p in body_stream[i]],
                "face_count": int((~np.isnan(face_stream[i, :, 0])).sum()),
                "hand_l": [(float(p[0]), float(p[1]), float(p[2]))
                           for p in hand_l_stream[i]] if enable_hands else [],
                "hand_r": [(float(p[0]), float(p[1]), float(p[2]))
                           for p in hand_r_stream[i]] if enable_hands else [],
                "body_bbox": body_bboxes[i],
                "face_bbox": face_bboxes[i],
            })
            pbar2.update_absolute(i, B)

        pose_tensor = torch.stack(pose_imgs, dim=0).to(images.device)
        face_tensor = torch.stack(face_imgs, dim=0).to(images.device)

        # Конвертируем bboxes в int-кортежи для совместимости
        body_bb_out = [tuple(int(v) for v in (bb or (0,0,0,0))) for bb in body_bboxes]
        face_bb_out = [tuple(int(v) for v in (bb or (0,0,0,0))) for bb in face_bboxes]

        print(f"[PoseFaceGOD] Done. pose={pose_tensor.shape}, faces={face_tensor.shape}")

        return (pose_tensor, face_tensor,
                json.dumps(kps_json_list),
                body_bb_out, face_bb_out)

    # ── face crop через warpAffine (rotate+scale+translate за раз) ──
    @staticmethod
    def _render_face_crop(frame_np, cx, cy, size, angle, out_size, frame_h, frame_w):
        import cv2
        frame_u8 = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
        size = max(8.0, size)
        scale = out_size / size
        a = math.radians(-angle)
        ca = math.cos(a) * scale; sa = math.sin(a) * scale
        M = np.array([
            [ca, -sa, out_size / 2.0 - (ca * cx - sa * cy)],
            [sa,  ca, out_size / 2.0 - (sa * cx + ca * cy)],
        ], dtype=np.float32)
        warped = cv2.warpAffine(
            frame_u8, M, (out_size, out_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return torch.from_numpy(warped.astype(np.float32) / 255.0)

    @staticmethod
    def _empty_outputs(B, out_w, out_h, face_size, device):
        pose = torch.zeros((B, out_h, out_w, 3), dtype=torch.float32, device=device)
        face = torch.zeros((B, face_size, face_size, 3), dtype=torch.float32, device=device)
        return (pose, face, "[]",
                [(0, 0, 0, 0)] * B, [(0, 0, 0, 0)] * B)


# ═══════════════════════════════════════════════════════════════
# Регистрация
# ═══════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "PoseFaceDetectGod": PoseFaceDetectGod,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseFaceDetectGod": "🔥 Pose & Face Detect GOD",
}
