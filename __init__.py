from .stable_face_crop import NODE_CLASS_MAPPINGS as A, NODE_DISPLAY_NAME_MAPPINGS as B
from .insightface_mouth_crop import NODE_CLASS_MAPPINGS as C, NODE_DISPLAY_NAME_MAPPINGS as D

try:
    from .pose_face_god import NODE_CLASS_MAPPINGS as E, NODE_DISPLAY_NAME_MAPPINGS as F
except Exception as _e:
    print(f"[comfyui-stable-face-crop] PoseFaceDetectGod not loaded: {_e}")
    E, F = {}, {}

try:
    from .pose_ultra_god import NODE_CLASS_MAPPINGS as G, NODE_DISPLAY_NAME_MAPPINGS as H
except Exception as _e:
    print(f"[comfyui-stable-face-crop] PoseUltraGod not loaded: {_e}")
    G, H = {}, {}

try:
    from .pose_extractor_nodes import NODE_CLASS_MAPPINGS as I, NODE_DISPLAY_NAME_MAPPINGS as J
except Exception as _e:
    print(f"[comfyui-stable-face-crop] PoseExtractor not loaded: {_e}")
    I, J = {}, {}

try:
    from .face_extractor_nodes import NODE_CLASS_MAPPINGS as K, NODE_DISPLAY_NAME_MAPPINGS as L
except Exception as _e:
    print(f"[comfyui-stable-face-crop] FaceExtractor not loaded: {_e}")
    K, L = {}, {}

NODE_CLASS_MAPPINGS = {**A, **C, **E, **G, **I, **K}
NODE_DISPLAY_NAME_MAPPINGS = {**B, **D, **F, **H, **J, **L}