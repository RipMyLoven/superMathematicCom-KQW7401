"""
PoseExtractor -> SAM Coords bridge node for ComfyUI.

Takes the POSEDATA output of `PoseAndFaceDetection` from
kijai/ComfyUI-WanAnimatePreprocess OR the POSE_KEYPOINT output of
DWPose / OpenPose preprocessor nodes (comfyui_controlnet_aux) and
converts selected keypoints into the STRING format consumed by
SAM2 / SAM3 video segmentation nodes (Sam2VideoSegmentationAddPoints,
Sam2Segmentation, EasySAM3, ...):

    coords_positive = '[{"x": 123, "y": 456}, ...]'
    coords_negative = '[{"x": 123, "y": 456}, ...]'


POSEDATA (WanAnimatePreprocess) is a Python dict produced by
`PoseAndFaceDetection.process`:

    {
        "retarget_image":      np.ndarray | None,
        "pose_metas":          [AAPoseMeta, ...],   # retargeted
        "refer_pose_meta":     dict | None,
        "pose_metas_original": [meta_dict, ...],    # one per frame
    }

Each `meta_dict` has:
    {
        "width": int, "height": int,
        "keypoints_body": [[x, y, c], ...],  # normalised 0..1
        "keypoints_face": [[x, y, c], ...],  # normalised 0..1
        ...
    }


POSE_KEYPOINT (DWPose / OpenPose) is a list, one dict per frame:

    [
        {
            "version": "...",
            "people": [
                {
                    "pose_keypoints_2d":      [x1,y1,c1, x2,y2,c2, ...],  # 18 pts
                    "face_keypoints_2d":      [x,y,c, ...],               # 68/70
                    "hand_left_keypoints_2d": [x,y,c, ...],
                    "hand_right_keypoints_2d":[x,y,c, ...],
                },
                ...
            ],
            "canvas_width":  int,
            "canvas_height": int,
        },
        ...
    ]

Coords there are in PIXEL space.


Body keypoint indices (OpenPose / DWPose 18-point convention, identical
for both inputs):
   0 nose       1 neck        2 r_shoulder  3 r_elbow   4 r_wrist
   5 l_shoulder 6 l_elbow     7 l_wrist     8 r_hip     9 r_knee
  10 r_ankle   11 l_hip      12 l_knee    13 l_ankle  14 r_eye
  15 l_eye     16 r_ear      17 l_ear
"""

import json


# ---------- presets ----------------------------------------------------------

BODY_PRESETS = {
    "none":           [],
    "nose":           [0],
    "head":           [0, 14, 15, 16, 17],
    "neck":           [1],
    "neck_shoulders": [1, 2, 5],        # neck + shoulders
    "neck_arms":      [1, 2, 3, 4, 5, 6, 7],  # neck + full arm chain (stops raised arms)
    "torso":          [1, 2, 5, 8, 11],
    "shoulders":      [2, 5],
    "hips":           [8, 11],
    "arms":           [2, 3, 4, 5, 6, 7],
    "legs":           [8, 9, 10, 11, 12, 13],
    "all_body":       list(range(18)),
}

PRESET_NAMES = list(BODY_PRESETS.keys())
FACE_PRESET_NAMES = ["none", "all_face"]

SOURCE_MODES = ["auto", "pose_data", "pose_keypoint"]

# Body keypoint indices that belong to the head region
_HEAD_KP_INDICES = [0, 14, 15, 16, 17]  # nose, r_eye, l_eye, r_ear, l_ear


# ---------- helpers ----------------------------------------------------------

def _parse_indices(s):
    """Parse a comma / space / semicolon separated list of ints. Empty -> []."""
    if not s:
        return []
    out = []
    for tok in s.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


def _kp_xyc(kp):
    """Return (x, y, c) from a keypoint that may be list/tuple/np.ndarray."""
    try:
        x = float(kp[0])
        y = float(kp[1])
        c = float(kp[2]) if len(kp) > 2 else 1.0
    except (TypeError, IndexError, ValueError):
        return None
    return x, y, c


def _is_normalised(keypoints):
    """Heuristic: if any coord > 1.5 we assume already in pixel space."""
    for kp in keypoints:
        xyc = _kp_xyc(kp)
        if xyc is None:
            continue
        x, y, _ = xyc
        if x > 1.5 or y > 1.5:
            return False
    return True


def _flat_to_xyc(flat):
    """Convert a flat [x,y,c,x,y,c,...] list (DWPose) into [[x,y,c],...]."""
    if flat is None:
        return []
    try:
        n = len(flat) // 3
    except TypeError:
        return []
    out = []
    for i in range(n):
        out.append([flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]])
    return out


def _collect_from_kp_list(keypoints, indices, min_conf, w, h, normalised):
    """Pick keypoints by index, drop low-confidence / missing, return [{x,y}]."""
    if keypoints is None or not indices:
        return []
    pts = []
    n = len(keypoints)
    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        xyc = _kp_xyc(keypoints[idx])
        if xyc is None:
            continue
        x, y, c = xyc
        if c < min_conf:
            continue
        if x == 0.0 and y == 0.0:  # missing
            continue
        if normalised:
            x *= w
            y *= h
        pts.append({"x": int(round(x)), "y": int(round(y))})
    return pts


def _compute_head_bbox(body, face, body_head_indices, w, h, normalised,
                       min_conf, hair_ext):
    """Compute a bounding box around the head + estimated hair region.

    Uses body head keypoints (nose, eyes, ears) and all face landmarks
    to determine the face bounding box, then extends it upward by
    ``hair_ext * face_height`` to cover the hair.

    Returns (x1, y1, x2, y2) in pixel space, or None if no points found.
    """
    pts = []
    if body and body_head_indices:
        n = len(body)
        for idx in body_head_indices:
            if not (0 <= idx < n):
                continue
            xyc = _kp_xyc(body[idx])
            if xyc is None:
                continue
            x, y, c = xyc
            if c < min_conf or (x == 0.0 and y == 0.0):
                continue
            if normalised:
                x *= w
                y *= h
            pts.append((x, y))
    if face:
        for kp in face:
            xyc = _kp_xyc(kp)
            if xyc is None:
                continue
            x, y, c = xyc
            if c < min_conf or (x == 0.0 and y == 0.0):
                continue
            if normalised:
                x *= w
                y *= h
            pts.append((x, y))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # extend upward for hair
    face_h = y2 - y1
    y1 = max(0.0, y1 - face_h * hair_ext)
    # clamp to canvas bounds
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(w - 1, int(round(x2))),
        min(h - 1, int(round(y2))),
    )


def _estimate_hair_ext(face, body, w, h, normalised, min_conf):
    """Estimate hair_extension ratio automatically from detected keypoints.

    The 68-point face model tracks up to the eyebrows (its topmost points).
    We use the nose body keypoint (index 0) as a reference: the nose is
    approximately 55-65 % of the way from the eyebrow line to the chin.
    The crown of the head is roughly 0.75× that same eyebrow-to-nose
    distance above the eyebrow line.  Adding a small hair margin gives a
    data-driven estimate that adapts to the detected face size.

    Returns a hair_ext float in [0.3, 1.2], or 0.55 as a safe default.
    """
    DEFAULT = 0.55
    if not face:
        return DEFAULT

    ys = []
    for kp in face:
        xyc = _kp_xyc(kp)
        if xyc is None:
            continue
        x, y, c = xyc
        if c < min_conf or (x == 0.0 and y == 0.0):
            continue
        ys.append(y * h if normalised else y)
    if not ys:
        return DEFAULT

    face_top_y = min(ys)          # ~ top of eyebrows
    face_h = max(ys) - face_top_y
    if face_h <= 0:
        return DEFAULT

    # Refine using nose body keypoint (index 0) for per-person calibration.
    if body and len(body) > 0:
        xyc = _kp_xyc(body[0])
        if xyc:
            nx, ny, nc = xyc
            if nc >= min_conf and not (nx == 0.0 and ny == 0.0):
                ny_px = ny * h if normalised else ny
                # distance from eyebrow line down to nose tip
                nose_below_top = ny_px - face_top_y
                if 0 < nose_below_top < face_h:
                    # crown ≈ 0.75 × (eyebrow-to-nose) above the eyebrow line
                    crown_offset = nose_below_top * 0.75
                    # small margin above crown for actual hair
                    hair_margin  = face_h * 0.15
                    ext = (crown_offset + hair_margin) / face_h
                    return max(0.3, min(1.2, ext))

    return DEFAULT


def _generate_head_frame_neg(bbox, w, h, margin_ratio=0.15, n_per_side=8,
                             bottom_rows=4, bottom_depth_ratio=2,
                             side_extra_ratio=0.5):
    """Negative point barrier around the head bbox (wide bottom + sides).

    Creates a thick U-shaped wall of SAM negative clicks just outside the
    detected head bounding box.  The bottom is filled with **multiple rows**
    of points extending downward by ``bottom_depth_ratio * face_height``, so
    SAM cannot grow the mask through a thin gap into the body/arms.
    Side columns are also widened outward by ``side_extra_ratio * face_width``.
    The top is intentionally left open so hair above the head is not blocked.
    """
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    face_h = max(1, y2 - y1)
    face_w = max(1, x2 - x1)
    margin = face_h * margin_ratio

    def _pt(px, py):
        return {"x": max(0, min(w - 1, int(round(px)))),
                "y": max(0, min(h - 1, int(round(py))))}

    result = []
    n = max(2, n_per_side)
    side_extra = face_w * side_extra_ratio

    # ---- BOTTOM WALL: multiple rows extending downward from chin --------
    bot_start = y2 + margin
    bot_end   = y2 + face_h * bottom_depth_ratio
    rows = max(1, bottom_rows)
    # widen the bottom wall horizontally so arms beside the head are blocked
    bot_x_left  = x1 - side_extra
    bot_x_right = x2 + side_extra
    bot_width   = bot_x_right - bot_x_left
    for r in range(rows):
        ty = r / max(1, rows - 1) if rows > 1 else 0
        cy = bot_start + ty * (bot_end - bot_start)
        for i in range(n):
            t = i / (n - 1)
            result.append(_pt(bot_x_left + t * bot_width, cy))

    # ---- LEFT WALL: column alongside head, plus an extra column further out
    for col_off in (margin, margin + side_extra * 0.6):
        cx_l = x1 - col_off
        for i in range(n):
            t = i / (n - 1)
            result.append(_pt(cx_l, y1 + t * face_h))

    # ---- RIGHT WALL: same on the right side
    for col_off in (margin, margin + side_extra * 0.6):
        cx_r = x2 + col_off
        for i in range(n):
            t = i / (n - 1)
            result.append(_pt(cx_r, y1 + t * face_h))

    return result


def _generate_hair_points(body, face, body_head_indices, w, h, normalised,
                          min_conf, hair_ext):
    """Generate sample points in the estimated hair region above the face.

    Scatters 5 points across the hair band (from hairline up to
    ``hair_ext * face_height`` above the top of the face bbox)
    so that SAM3 receives positive clicks inside the hair, not just on it.

    Returns a list of {"x": int, "y": int} dicts, or [] if the head
    region cannot be determined.
    """
    pts = []
    if body and body_head_indices:
        n = len(body)
        for idx in body_head_indices:
            if not (0 <= idx < n):
                continue
            xyc = _kp_xyc(body[idx])
            if xyc is None:
                continue
            x, y, c = xyc
            if c < min_conf or (x == 0.0 and y == 0.0):
                continue
            if normalised:
                x *= w
                y *= h
            pts.append((x, y))
    if face:
        for kp in face:
            xyc = _kp_xyc(kp)
            if xyc is None:
                continue
            x, y, c = xyc
            if c < min_conf or (x == 0.0 and y == 0.0):
                continue
            if normalised:
                x *= w
                y *= h
            pts.append((x, y))
    if not pts or hair_ext <= 0:
        return []

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1      = min(ys)   # top of detected face region
    face_w  = x2 - x1
    face_h  = max(ys) - y1
    if face_h <= 0:
        return []

    cx          = (x1 + x2) / 2.0
    half_spread = face_w * 0.35
    hair_top_y  = max(0.0, y1 - face_h * hair_ext)
    hair_mid_y  = (y1 + hair_top_y) / 2.0
    hairline_y  = max(0.0, y1 - face_h * 0.1)

    # 5 points: hairline, centre of hair band, left / right in band, near top
    candidates = [
        (cx,                 hairline_y),
        (cx,                 hair_mid_y),
        (cx - half_spread,   hair_mid_y),
        (cx + half_spread,   hair_mid_y),
        (cx,                 hair_top_y + face_h * 0.05),
    ]
    result = []
    for hx, hy in candidates:
        result.append({
            "x": max(0, min(w - 1, int(round(hx)))),
            "y": max(0, min(h - 1, int(round(hy)))),
        })
    return result


def _to_list(obj):
    """Convert np.ndarray / tuple to plain list (shallow)."""
    if obj is None:
        return None
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return list(obj) if not isinstance(obj, list) else obj


# ---------- POSEDATA (WanAnimatePreprocess) extraction ----------------------

def _frame_from_pose_data(pose_data, frame_index):
    """Return (body, face, w, h, normalised) for POSEDATA at frame_index."""
    metas = None
    if isinstance(pose_data, dict):
        metas = pose_data.get("pose_metas_original") \
             or pose_data.get("pose_metas")
    elif isinstance(pose_data, (list, tuple)):
        metas = pose_data

    if not metas:
        return None

    idx = max(0, min(int(frame_index), len(metas) - 1))
    meta = metas[idx]

    if not isinstance(meta, dict):
        meta = {
            "keypoints_body": getattr(meta, "keypoints_body", None),
            "keypoints_face": getattr(meta, "keypoints_face", None),
            "width":  getattr(meta, "width", 0),
            "height": getattr(meta, "height", 0),
        }

    body = _to_list(meta.get("keypoints_body"))
    face = _to_list(meta.get("keypoints_face"))
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)

    sample = body if body else face
    normalised = _is_normalised(sample) if sample else True

    return body, face, w, h, normalised


# ---------- POSE_KEYPOINT (DWPose) extraction -------------------------------

def _frame_from_pose_keypoint(pose_keypoint, frame_index, person_index):
    """Return (body, face, w, h, normalised) for POSE_KEYPOINT."""
    frames = pose_keypoint
    # Some nodes wrap it in a single dict instead of a list.
    if isinstance(frames, dict):
        frames = [frames]
    if not frames:
        return None

    idx = max(0, min(int(frame_index), len(frames) - 1))
    frame = frames[idx]
    if not isinstance(frame, dict):
        return None

    people = frame.get("people") or []
    if not people:
        body, face = [], []
    else:
        pidx = max(0, min(int(person_index), len(people) - 1))
        person = people[pidx] or {}
        body = _flat_to_xyc(person.get("pose_keypoints_2d"))
        face = _flat_to_xyc(person.get("face_keypoints_2d"))

    w = int(frame.get("canvas_width")  or 0)
    h = int(frame.get("canvas_height") or 0)

    # DWPose output is in pixel space.
    return body, face, w, h, False


# ---------- node -------------------------------------------------------------

class PoseDataToFaceSamCoords:
    """Convert pose keypoints (WanAnimate POSEDATA or DWPose POSE_KEYPOINT)
    into SAM2 / SAM3 coord strings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source":          (SOURCE_MODES, {"default": "auto"}),
                "frame_index":     ("INT",   {"default": 0, "min": 0, "max": 99999}),
                "positive_preset": (PRESET_NAMES, {"default": "head"}),
                "negative_preset": (PRESET_NAMES, {"default": "neck_arms"}),
                "face_preset":     (FACE_PRESET_NAMES, {"default": "none"}),
                "min_confidence":  ("FLOAT", {"default": 0.3, "min": 0.0,
                                              "max": 1.0, "step": 0.05}),
            },
            "optional": {
                # at least one of these should be connected
                "pose_data":     ("POSEDATA",),
                "pose_keypoint": ("POSE_KEYPOINT",),
                # which person to use from POSE_KEYPOINT (DWPose)
                "person_index":  ("INT", {"default": 0, "min": 0, "max": 64}),
                # extra custom indices (comma-separated) merged with presets
                "positive_body_idx": ("STRING", {"default": "", "multiline": False}),
                "positive_face_idx": ("STRING", {"default": "", "multiline": False}),
                "negative_body_idx": ("STRING", {"default": "", "multiline": False}),
                "negative_face_idx": ("STRING", {"default": "", "multiline": False}),
                # override output canvas size (0 -> take from pose source)
                "image_width":  ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
                # how far above the face to extend the head bbox to cover hair
                # (ratio of face height, e.g. 0.5 = extend by half the face height)
                # ignored when auto_hair_length is True
                "hair_extension": ("FLOAT", {"default": 0.5, "min": 0.0,
                                              "max": 3.0, "step": 0.05}),
                # when True, hair_extension is estimated automatically from
                # the detected face proportions (nose vs. eyebrow positions)
                "auto_hair_length": ("BOOLEAN", {"default": True}),
                # "auto": place sample points inside the hair region and add
                # them to coords_positive so SAM3 clicks inside the hair
                "hair_points": (["none", "auto"], {"default": "none"}),
                # auto-generate negative points around the head bbox (U-shape:
                # bottom + left + right sides) to prevent arm/body leakage
                "head_boundary": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT", "INT", "INT", "STRING", "BBOX", "BBOX")
    RETURN_NAMES = ("coords_positive", "coords_negative", "width", "height",
                    "head_x1", "head_y1", "head_x2", "head_y2",
                    "coords_hair", "head_bbox", "body_bbox")
    FUNCTION = "convert"
    CATEGORY = "PoseExtractor"

    def _pick_frame(self, source, pose_data, pose_keypoint,
                    frame_index, person_index):
        """Resolve `source` mode and return per-frame keypoints."""
        if source == "pose_data":
            if pose_data is None:
                return None
            return _frame_from_pose_data(pose_data, frame_index)

        if source == "pose_keypoint":
            if pose_keypoint is None:
                return None
            return _frame_from_pose_keypoint(pose_keypoint,
                                             frame_index, person_index)

        # auto: prefer whichever is connected; if both, prefer pose_data
        # (it carries the WanAnimate canvas size used downstream).
        if pose_data is not None:
            res = _frame_from_pose_data(pose_data, frame_index)
            if res is not None:
                return res
        if pose_keypoint is not None:
            return _frame_from_pose_keypoint(pose_keypoint,
                                             frame_index, person_index)
        return None

    def convert(self, source, frame_index,
                positive_preset, negative_preset, face_preset, min_confidence,
                pose_data=None, pose_keypoint=None, person_index=0,
                positive_body_idx="", positive_face_idx="",
                negative_body_idx="", negative_face_idx="",
                image_width=0, image_height=0, hair_extension=0.5,
                hair_points="none", auto_hair_length=True, head_boundary=True):

        picked = self._pick_frame(source, pose_data, pose_keypoint,
                                  frame_index, person_index)
        if picked is None:
            return ("[]", "[]", int(image_width), int(image_height), 0, 0, 0, 0, "[]", [], [])

        body, face, src_w, src_h, normalised = picked

        # canvas size: explicit override wins, else from source
        w = int(image_width)  or src_w
        h = int(image_height) or src_h
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1

        # ---- index sets --------------------------------------------------
        pos_body = list(BODY_PRESETS.get(positive_preset, [])) \
                   + _parse_indices(positive_body_idx)
        neg_body = list(BODY_PRESETS.get(negative_preset, [])) \
                   + _parse_indices(negative_body_idx)
        pos_face = _parse_indices(positive_face_idx)
        neg_face = _parse_indices(negative_face_idx)

        # if "all_face" preset selected, include every available face landmark
        if face_preset == "all_face" and face:
            pos_face = list(dict.fromkeys(list(range(len(face))) + pos_face))

        # de-duplicate while preserving order
        pos_body = list(dict.fromkeys(pos_body))
        neg_body = list(dict.fromkeys(neg_body))
        pos_face = list(dict.fromkeys(pos_face))
        neg_face = list(dict.fromkeys(neg_face))

        pos_pts = (_collect_from_kp_list(body, pos_body, min_confidence,
                                         w, h, normalised)
                   + _collect_from_kp_list(face, pos_face, min_confidence,
                                           w, h, normalised))
        neg_pts = (_collect_from_kp_list(body, neg_body, min_confidence,
                                         w, h, normalised)
                   + _collect_from_kp_list(face, neg_face, min_confidence,
                                           w, h, normalised))

        # head bounding box: face landmarks + body head kps + hair extension
        hair_ext = (_estimate_hair_ext(face, body, w, h, normalised, min_confidence)
                    if auto_hair_length
                    else float(hair_extension))
        bbox = _compute_head_bbox(body, face, _HEAD_KP_INDICES,
                                   w, h, normalised, min_confidence,
                                   hair_ext)
        hx1, hy1, hx2, hy2 = bbox if bbox is not None else (0, 0, 0, 0)

        # hair region sample points
        hair_pts = _generate_hair_points(body, face, _HEAD_KP_INDICES,
                                         w, h, normalised, min_confidence,
                                         hair_ext)
        if hair_points == "auto" and hair_pts:
            pos_pts = pos_pts + hair_pts

        # U-shaped negative frame just outside the head bbox
        # (bottom + left + right), independent of body keypoint detection
        if head_boundary:
            neg_pts = neg_pts + _generate_head_frame_neg(bbox, w, h)

        # BBOX-compatible output: [[x1, y1, x2, y2]] (native BBOX type for direct
        # connection to SAM3 / KJNodes bbox inputs — no String→BBox converter needed)
        head_bbox_list = [[hx1, hy1, hx2, hy2]] if bbox is not None else []

        # full body bbox (all 18 body keypoints, no hair extension)
        body_bbox_raw = _compute_head_bbox(body, None, list(range(18)),
                                           w, h, normalised, min_confidence, 0.0)
        body_bbox_list = ([[body_bbox_raw[0], body_bbox_raw[1],
                            body_bbox_raw[2], body_bbox_raw[3]]]
                          if body_bbox_raw is not None else [])

        return (json.dumps(pos_pts), json.dumps(neg_pts), w, h,
                hx1, hy1, hx2, hy2, json.dumps(hair_pts),
                head_bbox_list, body_bbox_list)


# ---------- registration -----------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PoseDataToFaceSamCoords": PoseDataToFaceSamCoords,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDataToFaceSamCoords": "Pose Data → Face SAM Coords",
}
