"""
PoseExtractor -> SAM Coords bridge node for ComfyUI.

Takes the POSEDATA output of `PoseAndFaceDetection` from
kijai/ComfyUI-WanAnimatePreprocess and converts selected keypoints into
the STRING format consumed by SAM2 / SAM3 video segmentation nodes
(Sam2VideoSegmentationAddPoints, Sam2Segmentation, EasySAM3, ...):

    coords_positive = '[{"x": 123, "y": 456}, ...]'
    coords_negative = '[{"x": 123, "y": 456}, ...]'

POSEDATA is a Python dict produced by `PoseAndFaceDetection.process`:

    {
        "retarget_image":      np.ndarray | None,
        "pose_metas":          [AAPoseMeta, ...],   # retargeted
        "refer_pose_meta":     dict | None,
        "pose_metas_original": [meta_dict, ...],    # one per frame
    }

Each `meta_dict` (from `load_pose_metas_from_kp2ds_seq`) has at least:
    {
        "width":           int,
        "height":          int,
        "keypoints_body":  [[x, y, c], ...]  # normalised 0..1
        "keypoints_face":  [[x, y, c], ...]  # normalised 0..1
        ...
    }

Body keypoint indices (OpenPose / DWPose 18-point convention used by
WanAnimate's ViTPose post-processing):
   0 nose       1 neck        2 r_shoulder  3 r_elbow   4 r_wrist
   5 l_shoulder 6 l_elbow     7 l_wrist     8 r_hip     9 r_knee
  10 r_ankle   11 l_hip      12 l_knee    13 l_ankle  14 r_eye
  15 l_eye     16 r_ear      17 l_ear
"""

import json


# ---------- presets ----------------------------------------------------------

BODY_PRESETS = {
    "none":      [],
    "nose":      [0],
    "head":      [0, 14, 15, 16, 17],
    "neck":      [1],
    "torso":     [1, 2, 5, 8, 11],
    "shoulders": [2, 5],
    "hips":      [8, 11],
    "arms":      [2, 3, 4, 5, 6, 7],
    "legs":      [8, 9, 10, 11, 12, 13],
    "all_body":  list(range(18)),
}

PRESET_NAMES = list(BODY_PRESETS.keys())


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


def _frame_meta(pose_data, frame_index):
    """Pick the per-frame meta dict from a POSEDATA payload.

    Falls back gracefully if the structure is slightly different.
    """
    metas = None
    if isinstance(pose_data, dict):
        # Prefer the raw (un-retargeted) per-frame dicts: they always carry
        # 'keypoints_body', 'keypoints_face', 'width', 'height'.
        metas = pose_data.get("pose_metas_original") \
             or pose_data.get("pose_metas")
    elif isinstance(pose_data, (list, tuple)):
        metas = pose_data

    if not metas:
        return None

    idx = max(0, min(int(frame_index), len(metas) - 1))
    meta = metas[idx]

    # AAPoseMeta object -> expose attributes as a dict.
    if not isinstance(meta, dict):
        meta = {
            "keypoints_body": getattr(meta, "keypoints_body", None),
            "keypoints_face": getattr(meta, "keypoints_face", None),
            "width":  getattr(meta, "width", 0),
            "height": getattr(meta, "height", 0),
        }
    return meta


# ---------- node -------------------------------------------------------------

class PoseDataToSamCoords:
    """Convert WanAnimatePreprocess POSEDATA into SAM2/SAM3 coord strings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data":       ("POSEDATA",),
                "frame_index":     ("INT",   {"default": 0, "min": 0, "max": 99999}),
                "positive_preset": (PRESET_NAMES, {"default": "head"}),
                "negative_preset": (PRESET_NAMES, {"default": "none"}),
                "min_confidence":  ("FLOAT", {"default": 0.3, "min": 0.0,
                                              "max": 1.0, "step": 0.05}),
            },
            "optional": {
                # extra custom indices (comma-separated) merged with presets
                "positive_body_idx": ("STRING", {"default": "", "multiline": False}),
                "positive_face_idx": ("STRING", {"default": "", "multiline": False}),
                "negative_body_idx": ("STRING", {"default": "", "multiline": False}),
                "negative_face_idx": ("STRING", {"default": "", "multiline": False}),
                # override output canvas size (0 -> take from pose_data)
                "image_width":  ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("coords_positive", "coords_negative", "width", "height")
    FUNCTION = "convert"
    CATEGORY = "PoseExtractor"

    def convert(self, pose_data, frame_index,
                positive_preset, negative_preset, min_confidence,
                positive_body_idx="", positive_face_idx="",
                negative_body_idx="", negative_face_idx="",
                image_width=0, image_height=0):

        meta = _frame_meta(pose_data, frame_index)
        if meta is None:
            return ("[]", "[]", int(image_width), int(image_height))

        body = _to_list(meta.get("keypoints_body"))
        face = _to_list(meta.get("keypoints_face"))

        # canvas size
        w = int(image_width) or int(meta.get("width") or 0)
        h = int(image_height) or int(meta.get("height") or 0)
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1

        # detect coord space from body keypoints (fall back to face)
        sample = body if body else face
        normalised = _is_normalised(sample) if sample else True

        # ---- index sets --------------------------------------------------
        pos_body = list(BODY_PRESETS.get(positive_preset, [])) \
                   + _parse_indices(positive_body_idx)
        neg_body = list(BODY_PRESETS.get(negative_preset, [])) \
                   + _parse_indices(negative_body_idx)
        pos_face = _parse_indices(positive_face_idx)
        neg_face = _parse_indices(negative_face_idx)

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

        return (json.dumps(pos_pts), json.dumps(neg_pts), w, h)


# ---------- registration -----------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PoseDataToSamCoords": PoseDataToSamCoords,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDataToSamCoords": "Pose Data → SAM Coords",
}
