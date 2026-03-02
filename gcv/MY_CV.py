import os
# These MUST be the first lines in the file
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - runtime fallback if torch import fails
    torch = None

# Dynamic Pathing to find your model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "apex_8n.pt"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "models", "apex_8n.pt"),
)
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), MODEL_CANDIDATES[-1])


def _assert_model_exists():
    if any(os.path.exists(path) for path in MODEL_CANDIDATES):
        return
    checked = ", ".join(MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model file not found. Checked: {checked}")


def _mps_is_available():
    if torch is None:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False

class GCVWorker:
    def __init__(self, width, height):
        _assert_model_exists()
        self.model = YOLO(MODEL_PATH)
        # Prefer MPS for speed when available; otherwise run on CPU.
        self.device = "mps" if _mps_is_available() else "cpu"
        try:
            self.model.to(self.device)
        except Exception:
            if self.device != "cpu":
                self.device = "cpu"
                self.model.to(self.device)
            else:
                raise
        
        self.width, self.height = width, height
        self.center_x, self.center_y = width // 2, height // 2
        self.fov_radius = 180
        self.min_box_height = 12
        self.max_box_area_ratio = 0.20
        
        # State for Smoothing
        self.smooth_x, self.smooth_y = self.center_x, self.center_y
        
        print(f"M4 LINEAR-SYNC ONLINE | Model: {MODEL_PATH} | Device: {self.device}")

    def _neutral_packet(self):
        return bytearray([1, 0, 0, 0, 0])

    def process(self, frame):
        # 1. FOV CIRCLE (Visualized)
        cv2.circle(frame, (self.center_x, self.center_y), self.fov_radius, (0, 255, 0), 2)

        # 2. WEAPON MASK (Hides gun to stop 'wonky' tracking)
        ai_frame = frame.copy()
        mask_x1, mask_y1 = int(self.width * 0.55), int(self.height * 0.55)
        cv2.rectangle(ai_frame, (mask_x1, mask_y1), (self.width, self.height), (0, 0, 0), -1)

        # 3. PREDICT (Non-streamed to avoid Error 121 crashes)
        try:
            results = self.model.predict(
                source=ai_frame, 
                conf=0.25, 
                imgsz=640, 
                device=self.device, 
                verbose=False
            )
        except Exception:
            self.smooth_x, self.smooth_y = self.center_x, self.center_y
            return (frame, self._neutral_packet())
        
        best_target = None
        min_dist = float('inf')

        boxes = None
        if results:
            first = results[0]
            boxes = getattr(first, "boxes", None)

        if boxes is not None and len(boxes) > 0:
            # Move data to CPU immediately to fix Error 121
            boxes = boxes.cpu() 
            for box in boxes:
                xyxy = box.xyxy[0].numpy()
                if not np.isfinite(xyxy).all():
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                if (w * h) > (self.width * self.height * self.max_box_area_ratio):
                    continue
                
                # Human Aspect Ratio Filter
                if (h / w) < 1.3 or h < self.min_box_height: continue

                # Snap to Upper Chest/Neck
                obj_x = int((x1 + x2) / 2)
                obj_y = int(y1 + h * 0.12) 

                dist = np.sqrt((obj_x - self.center_x)**2 + (obj_y - self.center_y)**2)
                
                if dist <= self.fov_radius and dist < min_dist:
                    min_dist = dist
                    best_target = (obj_x, obj_y, x1, y1, x2, y2)

        if best_target:
            tx, ty, x1, y1, x2, y2 = best_target
            # EMA Smoothing (0.40) - Essential for Linear curve stability
            self.smooth_x = (tx * 0.40) + (self.smooth_x * 0.60)
            self.smooth_y = (ty * 0.40) + (self.smooth_y * 0.60)
            
            # Draw Feedback
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(self.smooth_x), int(self.smooth_y)), 4, (0, 0, 255), -1)
        else:
            # Reset to center immediately when target is lost.
            self.smooth_x = self.center_x
            self.smooth_y = self.center_y

        # Rel x/y for Titan Two
        rel_x, rel_y = int(self.smooth_x - self.center_x), int(self.smooth_y - self.center_y)
        gcvdata = self._neutral_packet()
        gcvdata[1:3] = rel_x.to_bytes(2, byteorder='big', signed=True)
        gcvdata[3:5] = rel_y.to_bytes(2, byteorder='big', signed=True)
        
        return (frame, gcvdata)

    def __del__(self):
        pass
