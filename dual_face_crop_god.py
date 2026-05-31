import math
import torch
import numpy as np
import comfy.utils

# ─── Переиспользуем общий детектор из основного модуля ───
from .stable_face_crop import (
    get_face_detector,
    _interpolate_nans,
    _bidirectional_ema,
    _moving_average,
    _one_euro,
    _savgol,
    _deadzone,
    _velocity_clamp,
    _relative_velocity_clamp,
    _median_filter,
)


# ═══════════════════════════════════════════════════════════════
# Мульти-лицевая детекция: возвращает до N лиц.
# Каждый элемент — dict с bbox, landmarks, embedding (если есть).
# ═══════════════════════════════════════════════════════════════

def _empty_face_info():
    return {'bbox': None, 'landmarks': {}, 'all_found': False, 'face_ratio': 0.0, 'embedding': None}


def _dedup_faces(faces, iou_thresh=0.5):
    """Убирает дубликаты (перекрывающиеся bbox) — оставляет с максимальным det_score."""
    kept = []
    for f in sorted(faces, key=lambda x: -getattr(x, 'det_score', 0.0)):
        fx1, fy1, fx2, fy2 = f.bbox
        overlaps = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k.bbox
            ix1 = max(fx1, kx1); iy1 = max(fy1, ky1)
            ix2 = min(fx2, kx2); iy2 = min(fy2, ky2)
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            union = (fx2-fx1)*(fy2-fy1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter / union > iou_thresh:
                overlaps = True
                break
        if not overlaps:
            kept.append(f)
    return kept


def detect_all_faces(frame_np_uint8, detector, detector_type):
    """
    Возвращает список всех найденных лиц (неограниченное кол-во).
    Каждый элемент содержит bbox, landmarks, embedding (только insightface).
    Лица отсортированы по размеру bbox (от большего к меньшему).
    """
    h, w = frame_np_uint8.shape[:2]
    results = []

    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        min_face_px = max(30, int(min(h, w) * 0.04))

        def _face_to_info(face, x_offset=0):
            """Конвертирует insightface Face → info dict с явным смещением X."""
            info = _empty_face_info()
            x1 = int(face.bbox[0]) + x_offset
            y1 = int(face.bbox[1])
            x2 = int(face.bbox[2]) + x_offset
            y2 = int(face.bbox[3])
            info['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            bw2 = x2 - x1; bh2 = y2 - y1
            info['face_ratio'] = (bw2 * bh2) / max(1, w * h)
            if hasattr(face, 'embedding') and face.embedding is not None:
                norm = np.linalg.norm(face.embedding)
                info['embedding'] = (face.embedding / (norm + 1e-8)).copy()
            if face.kps is not None and len(face.kps) >= 5:
                kps = face.kps
                info['landmarks']['left_eye']    = (int(kps[0][0]) + x_offset, int(kps[0][1]))
                info['landmarks']['right_eye']   = (int(kps[1][0]) + x_offset, int(kps[1][1]))
                info['landmarks']['nose']        = (int(kps[2][0]) + x_offset, int(kps[2][1]))
                info['landmarks']['mouth_left']  = (int(kps[3][0]) + x_offset, int(kps[3][1]))
                info['landmarks']['mouth_right'] = (int(kps[4][0]) + x_offset, int(kps[4][1]))
                info['all_found'] = True
            return info

        def _no_overlap(info, existing, iou_thresh=0.4):
            ib = info['bbox']
            for r in existing:
                rb = r['bbox']
                ix1 = max(rb[0], ib[0]); iy1 = max(rb[1], ib[1])
                ix2 = min(rb[2], ib[2]); iy2 = min(rb[3], ib[3])
                iw2 = max(0, ix2 - ix1); ih2 = max(0, iy2 - iy1)
                inter = iw2 * ih2
                union = (rb[2]-rb[0])*(rb[3]-rb[1]) + (ib[2]-ib[0])*(ib[3]-ib[1]) - inter
                if union > 0 and inter / union > iou_thresh:
                    return False
            return True

        # ── Проход 1: стандартный порог по всему кадру ───────────
        faces = detector.get(bgr)
        valid1 = [f for f in faces
                  if (f.bbox[2]-f.bbox[0]) >= min_face_px
                  and (f.bbox[3]-f.bbox[1]) >= min_face_px
                  and getattr(f, 'det_score', 1.0) >= 0.45]
        valid1 = _dedup_faces(valid1, iou_thresh=0.5)
        for face in sorted(valid1, key=lambda f: -((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))):
            results.append(_face_to_info(face, x_offset=0))

        # ── Проход 2: ищем в противоположной половине кадра ──────
        if len(results) < 2:
            if results:
                known_cx = (results[0]['bbox'][0] + results[0]['bbox'][2]) / 2.0
                if known_cx > w / 2:
                    # первое лицо справа → ищем в левой части
                    x_end   = max(int(known_cx - w * 0.05), int(w * 0.15))
                    half_bgr = bgr[:, :x_end]
                    offset_x = 0
                else:
                    # первое лицо слева → ищем в правой части
                    x_start  = min(int(known_cx + w * 0.05), int(w * 0.85))
                    half_bgr = bgr[:, x_start:]
                    offset_x = x_start
            else:
                # лиц не найдено вообще — пробуем весь кадр с низким порогом
                half_bgr = bgr
                offset_x = 0

            if half_bgr.shape[1] >= min_face_px * 2:
                _old_thresh = getattr(detector, 'det_thresh', 0.5)
                try:
                    detector.det_thresh = 0.3
                except Exception:
                    pass
                try:
                    extra_raw = detector.get(half_bgr)
                except Exception:
                    extra_raw = []
                try:
                    detector.det_thresh = _old_thresh
                except Exception:
                    pass

                for f in sorted(extra_raw, key=lambda x: -getattr(x, 'det_score', 0.0)):
                    bw2 = f.bbox[2] - f.bbox[0]; bh2 = f.bbox[3] - f.bbox[1]
                    if (bw2 < min_face_px or bh2 < min_face_px
                            or getattr(f, 'det_score', 0.0) < 0.25
                            or f.kps is None or len(f.kps) < 5):
                        continue
                    info = _face_to_info(f, x_offset=offset_x)
                    if _no_overlap(info, results, iou_thresh=0.4):
                        results.append(info)
                        print(f"[DUAL-GOD] half-search: face2 bbox="
                              f"{info['bbox']} offset={offset_x} "
                              f"score={getattr(f,'det_score',0):.2f}")
                        if len(results) >= 2:
                            break

    elif detector_type == "mediapipe":
        det_results = detector.process(frame_np_uint8)
        if det_results.detections:
            for det in det_results.detections:
                info = _empty_face_info()
                bb = det.location_data.relative_bounding_box
                x1 = int(bb.xmin * w);                  y1 = int(bb.ymin * h)
                x2 = int((bb.xmin + bb.width) * w);     y2 = int((bb.ymin + bb.height) * h)
                info['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                info['face_ratio'] = bb.width * bb.height
                kp = det.location_data.relative_keypoints
                if len(kp) >= 4:
                    info['landmarks']['right_eye'] = (int(kp[0].x * w), int(kp[0].y * h))
                    info['landmarks']['left_eye']  = (int(kp[1].x * w), int(kp[1].y * h))
                    info['landmarks']['nose']      = (int(kp[2].x * w), int(kp[2].y * h))
                    mx = int(kp[3].x * w); my = int(kp[3].y * h)
                    ed = abs(info['landmarks']['left_eye'][0] - info['landmarks']['right_eye'][0])
                    hm = max(ed // 3, 10)
                    info['landmarks']['mouth_left']  = (mx - hm, my)
                    info['landmarks']['mouth_right'] = (mx + hm, my)
                    info['all_found'] = True
                results.append(info)
            results = sorted(results, key=lambda r: -r['face_ratio'])

    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2GRAY)
        faces_detected = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces_detected) > 0:
            for fx, fy, fw, fh in faces_detected:
                info = _empty_face_info()
                info['bbox'] = (fx, fy, fx + fw, fy + fh)
                info['face_ratio'] = (fw * fh) / max(1, w * h)
                results.append(info)
            results = sorted(results, key=lambda r: -r['face_ratio'])

    return results


def detect_two_faces_full(frame_np_uint8, detector, detector_type):
    """
    Возвращает список из 2 элементов (каждый — face_info dict).
    Если найдено менее 2 лиц — остальные slots заполнены _empty_face_info().
    Лица отсортированы по x-центру: [0] = левее, [1] = правее.
    """
    h, w = frame_np_uint8.shape[:2]
    results = []

    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        faces = detector.get(bgr)
        # Сортируем по x-центру лица
        faces = sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2.0)
        for face in faces[:2]:
            info = _empty_face_info()
            x1, y1, x2, y2 = face.bbox.astype(int)
            bw = x2 - x1; bh = y2 - y1
            info['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            info['face_ratio'] = (bw * bh) / max(1, w * h)
            if face.kps is not None and len(face.kps) >= 5:
                kps = face.kps.astype(int)
                info['landmarks']['left_eye']    = tuple(kps[0])
                info['landmarks']['right_eye']   = tuple(kps[1])
                info['landmarks']['nose']        = tuple(kps[2])
                info['landmarks']['mouth_left']  = tuple(kps[3])
                info['landmarks']['mouth_right'] = tuple(kps[4])
                info['all_found'] = True
            results.append(info)

    elif detector_type == "mediapipe":
        det_results = detector.process(frame_np_uint8)
        if det_results.detections:
            # Сортируем по x-центру
            dets = sorted(
                det_results.detections,
                key=lambda d: d.location_data.relative_bounding_box.xmin
                              + d.location_data.relative_bounding_box.width / 2.0
            )
            for det in dets[:2]:
                info = _empty_face_info()
                bb = det.location_data.relative_bounding_box
                x1 = int(bb.xmin * w);                  y1 = int(bb.ymin * h)
                x2 = int((bb.xmin + bb.width) * w);     y2 = int((bb.ymin + bb.height) * h)
                info['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                info['face_ratio'] = bb.width * bb.height
                kp = det.location_data.relative_keypoints
                if len(kp) >= 4:
                    info['landmarks']['right_eye'] = (int(kp[0].x * w), int(kp[0].y * h))
                    info['landmarks']['left_eye']  = (int(kp[1].x * w), int(kp[1].y * h))
                    info['landmarks']['nose']      = (int(kp[2].x * w), int(kp[2].y * h))
                    mx = int(kp[3].x * w); my = int(kp[3].y * h)
                    ed = abs(info['landmarks']['left_eye'][0] - info['landmarks']['right_eye'][0])
                    hm = max(ed // 3, 10)
                    info['landmarks']['mouth_left']  = (mx - hm, my)
                    info['landmarks']['mouth_right'] = (mx + hm, my)
                    info['all_found'] = True
                results.append(info)

    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2GRAY)
        faces_detected = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces_detected) > 0:
            # Сортируем по x-центру
            faces_s = sorted(faces_detected, key=lambda f: f[0] + f[2] / 2.0)
            for fx, fy, fw, fh in faces_s[:2]:
                info = _empty_face_info()
                info['bbox'] = (fx, fy, fx + fw, fy + fh)
                info['face_ratio'] = (fw * fh) / max(1, w * h)
                results.append(info)

    # Дополняем до 2 слотов
    while len(results) < 2:
        results.append(_empty_face_info())

    return results  # всегда длина == 2


# ═══════════════════════════════════════════════════════════════
# НОДА: Dual Face Crop GOD
# ═══════════════════════════════════════════════════════════════

class DualFaceCropGod:
    """
    Детектирует 2 лица в видео и выдаёт два отдельных стабилизированных
    кропа с полным GOD-пайплайном (One Euro + bidir EMA + Savgol +
    velocity clamp + optical-flow fallback + warpAffine).

    Лицо 1 — левее в кадре, Лицо 2 — правее в кадре.
    Если в кадре одно лицо — второй выход даёт центр-кроп.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Ширина выхода для каждого лица"
                }),
                "output_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Высота выхода для каждого лица"
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
                    "tooltip": "0.5 = центр, <0.5 = влево, >0.5 = вправо"
                }),
                "enable_rotation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Авто-выравнивание по линии глаз"
                }),
                "max_rotation_deg": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Макс. угол поворота"
                }),
                "rotation_smoothing": ("FLOAT", {
                    "default": 0.92, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA для угла поворота"
                }),
                "position_smoothing": ("FLOAT", {
                    "default": 0.88, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA для позиции cx/cy"
                }),
                "size_smoothing": ("FLOAT", {
                    "default": 0.96, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA для размера/зума"
                }),
                "lock_size_to_median": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Фиксирует размер кропа на медиане — нулевое зум-дыхание"
                }),
                "pre_median_window": ("INT", {
                    "default": 5, "min": 0, "max": 31, "step": 1,
                    "tooltip": "Медианный фильтр ДО сглаживания. 0=выкл"
                }),
                "one_euro_min_cutoff": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "One Euro: ниже = плавнее в покое"
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
                    "tooltip": "Макс. сдвиг центра за кадр (0=выкл)"
                }),
                "max_zoom_per_frame": ("FLOAT", {
                    "default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Макс. относительное изменение размера за кадр (0=выкл)"
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
                    "tooltip": "Phase correlation трекинг если детектор молчит"
                }),
                "border_mode": (["reflect", "replicate", "black", "wrap"], {
                    "default": "reflect",
                    "tooltip": "Заполнение при выходе кропа за кадр"
                }),
                "resolution_divider": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Делитель разрешения (1.0=полное)"
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("face_1", "face_2")
    FUNCTION      = "process"
    CATEGORY      = "face/lipsync"
    DESCRIPTION   = (
        "🔥 Dual Face Crop GOD: детектирует 2 лица в видео и выдаёт два "
        "отдельных стабилизированных кропа. Лицо 1 = левее, Лицо 2 = правее."
    )

    # ── вспомогалки ──────────────────────────────────────────────

    @staticmethod
    def _angle_from_eyes(le, re):
        if le is None or re is None:
            return 0.0
        dx = re[0] - le[0]; dy = re[1] - le[1]
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _phase_shift(prev_gray, cur_gray):
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

    # ── ядро: один проход стабилизации для одного слота лица ─────

    def _stabilize_slot(self, raw_cx, raw_cy, raw_size, raw_angle,
                        detected_mask,
                        position_smoothing, size_smoothing, rotation_smoothing,
                        pre_median_window, one_euro_min_cutoff, one_euro_beta,
                        deadzone_pixels, max_velocity_px, max_zoom_per_frame,
                        size_iqr_clamp, savgol_window,
                        lock_size_to_median, enable_rotation, slot_name):
        B = len(raw_cx)
        n_det = int(detected_mask.sum())

        if n_det == 0:
            return None, None, None, None  # сигнал fallback

        raw_cx    = _interpolate_nans(raw_cx.copy())
        raw_cy    = _interpolate_nans(raw_cy.copy())
        raw_size  = _interpolate_nans(raw_size.copy())
        if enable_rotation:
            if np.all(np.isnan(raw_angle)):
                raw_angle = np.zeros(B)
            else:
                raw_angle = _interpolate_nans(raw_angle.copy())
        else:
            raw_angle = np.zeros(B)

        # Медианный пре-фильтр
        if pre_median_window and pre_median_window >= 3:
            mw = pre_median_window if pre_median_window % 2 == 1 else pre_median_window + 1
            raw_cx    = _median_filter(raw_cx,   mw)
            raw_cy    = _median_filter(raw_cy,   mw)
            raw_size  = _median_filter(raw_size, mw)
            if enable_rotation:
                raw_angle = _median_filter(raw_angle, mw)

        # IQR clamp
        if size_iqr_clamp > 0 and n_det >= 4:
            med = float(np.median(raw_size))
            raw_size = np.clip(raw_size,
                               med * (1.0 - size_iqr_clamp),
                               med * (1.0 + size_iqr_clamp))

        # Lock size
        if lock_size_to_median:
            locked = float(np.median(raw_size))
            raw_size = np.full(B, locked, dtype=np.float64)
            print(f"[DUAL-GOD][{slot_name}] 🔒 Size LOCKED = {locked:.1f}px")

        # One Euro
        sm_cx    = _one_euro(raw_cx,   min_cutoff=one_euro_min_cutoff,       beta=one_euro_beta)
        sm_cy    = _one_euro(raw_cy,   min_cutoff=one_euro_min_cutoff,       beta=one_euro_beta)
        sm_size  = _one_euro(raw_size, min_cutoff=one_euro_min_cutoff * 0.7, beta=one_euro_beta * 0.5)
        sm_angle = (_one_euro(raw_angle, min_cutoff=one_euro_min_cutoff * 0.5, beta=one_euro_beta * 0.3)
                    if enable_rotation else raw_angle)

        # Bidirectional EMA
        sm_cx    = _bidirectional_ema(sm_cx,   position_smoothing)
        sm_cy    = _bidirectional_ema(sm_cy,   position_smoothing)
        sm_size  = _bidirectional_ema(sm_size, size_smoothing)
        if enable_rotation:
            sm_angle = _bidirectional_ema(sm_angle, rotation_smoothing)

        # Savitzky-Golay
        if savgol_window and savgol_window >= 5:
            w = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
            sm_cx   = _savgol(sm_cx,   w, 3)
            sm_cy   = _savgol(sm_cy,   w, 3)
            sm_size = _savgol(sm_size, w, 2)
            if enable_rotation:
                sm_angle = _savgol(sm_angle, w, 2)

        # Deadzone
        if deadzone_pixels > 0:
            sm_cx = _deadzone(sm_cx, deadzone_pixels)
            sm_cy = _deadzone(sm_cy, deadzone_pixels)

        # Velocity clamp
        if max_velocity_px > 0:
            sm_cx = _velocity_clamp(sm_cx, max_velocity_px)
            sm_cy = _velocity_clamp(sm_cy, max_velocity_px)
        if max_zoom_per_frame > 0 and not lock_size_to_median:
            sm_size = _relative_velocity_clamp(sm_size, max_zoom_per_frame)
        if enable_rotation and max_velocity_px > 0:
            sm_angle = _velocity_clamp(sm_angle, 5.0)

        # Если lock — финально принудительно медиана
        if lock_size_to_median:
            sm_size = np.full(B, float(np.median(raw_size)), dtype=np.float64)

        return sm_cx, sm_cy, sm_size, sm_angle

    # ── рендер одного кропа ──────────────────────────────────────

    def _render_slot(self, images, sm_cx, sm_cy, sm_size, sm_angle,
                     shift_vertical, shift_horizontal,
                     aspect, enable_rotation, final_w, final_h, border, frames_u8):
        import cv2
        B = len(sm_cx)
        result_frames = []

        h_shift_arr = (shift_horizontal - 0.5) * sm_size * 0.4
        v_shift_arr = (shift_vertical   - 0.5) * sm_size * 0.4

        for i in range(B):
            cx   = float(sm_cx[i]  + h_shift_arr[i])
            cy   = float(sm_cy[i]  - v_shift_arr[i])
            size = float(max(8.0,   sm_size[i]))
            ang  = float(sm_angle[i]) if enable_rotation else 0.0

            if aspect >= 1.0:
                crop_w = size * aspect; crop_h = size
            else:
                crop_w = size; crop_h = size / aspect

            scale = final_w / crop_w

            try:
                frame_u8 = frames_u8[i]
                if frame_u8 is None:
                    # safeguard: всегда рендерим из оригинала
                    frame_u8 = (images[i].cpu().numpy() * 255).astype(np.uint8)
                    frames_u8[i] = frame_u8
                a  = math.radians(-ang) if enable_rotation else 0.0
                ca = math.cos(a) * scale
                sa = math.sin(a) * scale
                M  = np.array([
                    [ca,  -sa, final_w / 2.0 - (ca * cx  - sa * cy)],
                    [sa,   ca, final_h / 2.0 - (sa * cx  + ca * cy)],
                ], dtype=np.float32)
                warped = cv2.warpAffine(
                    frame_u8, M, (final_w, final_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=border,
                    borderValue=(0, 0, 0))
                t = torch.from_numpy(warped.astype(np.float32) / 255.0)
                result_frames.append(t)
            except Exception as e:
                print(f"[DUAL-GOD] frame {i} warp failed: {e}")
                result_frames.append(
                    self._fallback_crop(images[i], cx, cy, crop_w, crop_h, final_w, final_h))

        return torch.stack(result_frames, dim=0)

    @staticmethod
    def _fallback_crop(frame, cx, cy, cw, ch, out_w, out_h):
        H, W = frame.shape[:2]
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
        else:          crop_w = W; crop_h = int(W / ar)
        y1 = (H - crop_h) // 2; x1 = (W - crop_w) // 2
        c = images[:, y1:y1+crop_h, x1:x1+crop_w, :].permute(0, 3, 1, 2)
        r = torch.nn.functional.interpolate(c, size=(out_h, out_w),
                                             mode='bilinear', align_corners=False)
        return r.permute(0, 2, 3, 1)

    # ── identity matching ─────────────────────────────────────────

    @staticmethod
    def _face_similarity(fi, slot_emb, slot_cx, slot_cy, slot_rawsize):
        """
        Возвращает score сходства [0..1] между детекцией и слотом.
        Если есть embedding — используем косинусное сходство.
        Иначе — нормализованная дистанция центров.
        """
        bbox = fi.get('bbox')
        if bbox is None:
            return 0.0
        fcx = (bbox[0] + bbox[2]) / 2.0
        fcy = (bbox[1] + bbox[3]) / 2.0

        # Embedding score (insightface)
        emb = fi.get('embedding')
        if emb is not None and slot_emb is not None:
            cos_sim = float(np.dot(emb, slot_emb))  # оба уже нормализованы
            return max(0.0, cos_sim)

        # Fallback: proximity score
        if slot_cx is None or slot_rawsize is None or slot_rawsize <= 0:
            return 0.5  # нет истории — нейтральный score
        dist = math.sqrt((fcx - slot_cx)**2 + (fcy - slot_cy)**2)
        norm_dist = dist / max(slot_rawsize, 1.0)
        return max(0.0, 1.0 - norm_dist / 2.0)  # gate: за 2 размера лица = 0

    @staticmethod
    def _assign_faces_to_slots(detections, slot_emb, slot_cx, slot_cy, slot_rawsize,
                                min_similarity=0.25):
        """
        Назначает детекции слотам по similarity.
        Возвращает [(slot_idx, face_info), ...] только для назначенных пар.
        Одна детекция → один слот, один слот → одна детекция.
        """
        n_det = len(detections)
        if n_det == 0:
            return []

        known = [s for s in range(2) if slot_cx[s] is not None]

        if not known:
            # Первый кадр — инициализация по X: левое→0, правое→1
            sorted_by_x = sorted(enumerate(detections),
                                  key=lambda t: (t[1]['bbox'][0] + t[1]['bbox'][2]) / 2.0)
            return [(slot, sorted_by_x[slot][1]) for slot in range(min(2, len(sorted_by_x)))]

        # Строим матрицу score[det_idx][slot_idx]
        score = np.zeros((n_det, 2))
        for di, fi in enumerate(detections):
            for s in range(2):
                score[di][s] = DualFaceCropGod._face_similarity(
                    fi, slot_emb[s], slot_cx[s], slot_cy[s], slot_rawsize[s])

        # Назначаем жадно: берём лучший (det, slot) с учётом min_similarity
        assigned_det  = set()
        assigned_slot = set()
        result = []

        # Все возможные пары, отсортированные по score убыванию
        pairs = sorted(
            [(score[di][s], di, s) for di in range(n_det) for s in range(2)],
            reverse=True
        )
        for sc, di, s in pairs:
            if di in assigned_det or s in assigned_slot:
                continue
            if sc < min_similarity:
                break
            result.append((s, detections[di]))
            assigned_det.add(di)
            assigned_slot.add(s)

        # Fallback: если есть неинициализированный слот и остались несвязанные детекции —
        # назначаем самую далёкую от уже занятых слотов детекцию в пустой слот.
        unassigned_slots = [s for s in range(2) if s not in assigned_slot]
        unassigned_dets  = [di for di in range(n_det) if di not in assigned_det]
        for s in unassigned_slots:
            if not unassigned_dets:
                break
            if slot_cx[s] is None:  # слот ещё не инициализирован
                # Выбираем детекцию, максимально отличную по X от уже занятых слотов
                occupied_x = []
                for (rs, rfi) in result:
                    bb = rfi.get('bbox')
                    if bb:
                        occupied_x.append((bb[0] + bb[2]) / 2.0)
                def _score_for_uninit(di):
                    bb = detections[di].get('bbox')
                    if bb is None:
                        return -1.0
                    fx = (bb[0] + bb[2]) / 2.0
                    if not occupied_x:
                        return 0.0
                    return min(abs(fx - ox) for ox in occupied_x)
                best_di = max(unassigned_dets, key=_score_for_uninit)
                result.append((s, detections[best_di]))
                unassigned_dets.remove(best_di)

        return result

    # ── главный process ───────────────────────────────────────────

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
                border_mode="reflect", resolution_divider=1.0):

        import cv2
        B, H, W, C = images.shape
        final_w = max(64, (int(output_width  / resolution_divider) // 8) * 8)
        final_h = max(64, (int(output_height / resolution_divider) // 8) * 8)
        aspect  = output_width / max(1, output_height)

        print(f"[DUAL-GOD] {B} frames ({W}x{H}) → {final_w}x{final_h} "
              f"(AR={aspect:.2f}, rot={enable_rotation}, flow={optical_flow_fallback})")

        try:
            detector, detector_type = get_face_detector()
        except Exception as e:
            print(f"[DUAL-GOD] Detector failed: {e}. Center crop fallback.")
            cc = self._center_crop_batch(images, final_w, final_h)
            return (cc, cc)

        border_cv = {
            "reflect":   cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "black":     cv2.BORDER_CONSTANT,
            "wrap":      cv2.BORDER_WRAP,
        }.get(border_mode, cv2.BORDER_REFLECT_101)

        pbar = comfy.utils.ProgressBar(B)

        # Два слота: 0 = лицо-1, 1 = лицо-2
        raw_cx    = [np.full(B, np.nan), np.full(B, np.nan)]
        raw_cy    = [np.full(B, np.nan), np.full(B, np.nan)]
        raw_size  = [np.full(B, np.nan), np.full(B, np.nan)]
        raw_angle = [np.full(B, np.nan), np.full(B, np.nan)]
        det_mask  = [np.zeros(B, dtype=bool), np.zeros(B, dtype=bool)]

        # Optical-flow кэши
        prev_patch  = [None, None]
        prev_anchor = [None, None]  # (cx, cy, size)

        # Identity state для каждого слота
        slot_emb     = [None, None]   # усреднённый embedding (insightface)
        slot_cx      = [None, None]   # последний известный центр X
        slot_cy      = [None, None]   # последний известный центр Y
        slot_rawsize = [None, None]   # последний raw размер bbox (без padding)

        frames_u8 = [None] * B

        def _frame_u8(i):
            if frames_u8[i] is None:
                frames_u8[i] = (images[i].cpu().numpy() * 255).astype(np.uint8)
            return frames_u8[i]

        # ═══ ГЛАВНЫЙ ПРОХОД: детекция + identity tracking + optical flow ═══
        for i in range(B):
            do_detect = (i % detect_every_n == 0)
            matched_slots = set()

            if do_detect:
                try:
                    detections = detect_all_faces(_frame_u8(i), detector, detector_type)
                except Exception as exc:
                    print(f"[DUAL-GOD] detect error f{i}: {exc}")
                    detections = []

                if i % max(1, B // 10) == 0 or len(detections) != 2:
                    print(f"[DUAL-GOD] f{i}: found {len(detections)} face(s), "
                          f"slots init: {[slot_cx[s] is not None for s in range(2)]}")

                if detections:
                    matched = self._assign_faces_to_slots(
                        detections, slot_emb, slot_cx, slot_cy, slot_rawsize)

                    for slot, fi in matched:
                        bbox = fi.get('bbox')
                        if bbox is None:
                            continue
                        bw = bbox[2] - bbox[0]; bh = bbox[3] - bbox[1]
                        if bw <= 0 or bh <= 0:
                            continue

                        raw_face = max(bw, bh)
                        fsize    = raw_face * scale_padding
                        fcx      = (bbox[0] + bbox[2]) / 2.0
                        fcy      = (bbox[1] + bbox[3]) / 2.0
                        fang     = 0.0
                        if enable_rotation:
                            lm   = fi.get('landmarks', {})
                            a    = self._angle_from_eyes(lm.get('left_eye'), lm.get('right_eye'))
                            fang = float(np.clip(a, -max_rotation_deg, max_rotation_deg))

                        raw_cx   [slot][i] = fcx
                        raw_cy   [slot][i] = fcy
                        raw_size [slot][i] = fsize
                        raw_angle[slot][i] = fang
                        det_mask [slot][i] = True

                        # Обновляем identity state
                        slot_cx     [slot] = fcx
                        slot_cy     [slot] = fcy
                        slot_rawsize[slot] = raw_face
                        prev_anchor [slot] = (fcx, fcy, fsize)
                        prev_patch  [slot] = self._safe_gray_patch(
                            _frame_u8(i), fcx, fcy, fsize, target=128)

                        # Обновляем embedding — EMA для стабильности
                        emb = fi.get('embedding')
                        if emb is not None:
                            if slot_emb[slot] is None:
                                slot_emb[slot] = emb.copy()
                            else:
                                # EMA 0.7 старый + 0.3 новый — плавное обновление
                                slot_emb[slot] = 0.7 * slot_emb[slot] + 0.3 * emb
                                norm = np.linalg.norm(slot_emb[slot])
                                if norm > 1e-8:
                                    slot_emb[slot] /= norm

                        matched_slots.add(slot)
                        print(f"[DUAL-GOD] f{i} slot{slot} → face cx={fcx:.0f} cy={fcy:.0f} "
                              f"raw={raw_face:.0f}px emb={'yes' if emb is not None else 'no'}")

            # Optical flow для всех незафиксированных слотов (включая пропущенные кадры)
            for slot in range(2):
                if slot not in matched_slots:
                    if optical_flow_fallback and prev_patch[slot] is not None and prev_anchor[slot] is not None:
                        pcx, pcy, psize = prev_anchor[slot]
                        cp = self._safe_gray_patch(_frame_u8(i), pcx, pcy, psize, target=128)
                        shift = self._phase_shift(prev_patch[slot], cp)
                        if shift is not None and cp is not None:
                            scale_back = psize / 128.0
                            lim = psize * 0.3
                            dx = float(np.clip(shift[0] * scale_back, -lim, lim))
                            dy = float(np.clip(shift[1] * scale_back, -lim, lim))
                            ncx = pcx + dx; ncy = pcy + dy
                            raw_cx  [slot][i] = ncx
                            raw_cy  [slot][i] = ncy
                            raw_size[slot][i] = psize
                            det_mask[slot][i] = True
                            prev_anchor[slot] = (ncx, ncy, psize)
                            prev_patch [slot] = self._safe_gray_patch(
                                _frame_u8(i), ncx, ncy, psize, target=128)

            pbar.update_absolute(i, B)

        # Гарантируем что ВСЕ frames_u8 заполнены из оригинала перед рендером
        for i in range(B):
            if frames_u8[i] is None:
                frames_u8[i] = (images[i].cpu().numpy() * 255).astype(np.uint8)

        # Статистика
        for slot in range(2):
            nd = int(det_mask[slot].sum())
            print(f"[DUAL-GOD][face_{slot+1}] Tracked {nd}/{B} frames ({100.0*nd/max(1,B):.1f}%)")

        # Стабилизация и рендер каждого слота
        outputs = []
        slot_names = ["face_1", "face_2"]

        for slot in range(2):
            sm_cx, sm_cy, sm_size, sm_angle = self._stabilize_slot(
                raw_cx[slot], raw_cy[slot], raw_size[slot], raw_angle[slot],
                det_mask[slot],
                position_smoothing, size_smoothing, rotation_smoothing,
                pre_median_window, one_euro_min_cutoff, one_euro_beta,
                deadzone_pixels, max_velocity_px, max_zoom_per_frame,
                size_iqr_clamp, savgol_window,
                lock_size_to_median, enable_rotation, slot_names[slot]
            )

            if sm_cx is None:
                print(f"[DUAL-GOD][{slot_names[slot]}] No face detected — center crop fallback")
                outputs.append(self._center_crop_batch(images, final_w, final_h))
                continue

            rendered = self._render_slot(
                images, sm_cx, sm_cy, sm_size, sm_angle,
                shift_vertical, shift_horizontal,
                aspect, enable_rotation, final_w, final_h, border_cv, frames_u8
            )
            outputs.append(rendered.to(images.device))
            print(f"[DUAL-GOD][{slot_names[slot]}] Done. {rendered.shape}")

        return (outputs[0], outputs[1])


# ═══════════════════════════════════════════════════════════════
# Регистрация
# ═══════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "DualFaceCropGod": DualFaceCropGod,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualFaceCropGod": "👥 Dual Face Crop GOD",
}
