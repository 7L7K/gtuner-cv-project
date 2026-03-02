import os
# These MUST be the first lines in the file
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import cv2
import numpy as np
from ultralytics import YOLO

# Dynamic Pathing to find your model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "apex_8n.pt")

class GCVWorker:
    def __init__(self, width, height):
        self.model = YOLO(MODEL_PATH)
        # Load to MPS (GPU) for raw speed
        self.model.to('mps') 
        
        self.width, self.height = width, height
        self.center_x, self.center_y = width // 2, height // 2
        
        # State for Smoothing
        self.smooth_x, self.smooth_y = self.center_x, self.center_y
        
        print(f"M4 LINEAR-SYNC ONLINE | Model: {MODEL_PATH}")

    def process(self, frame):
        # 1. FOV CIRCLE (Visualized)
        cv2.circle(frame, (self.center_x, self.center_y), 180, (0, 255, 0), 2)

        # 2. WEAPON MASK (Hides gun to stop 'wonky' tracking)
        ai_frame = frame.copy()
        mask_x1, mask_y1 = int(self.width * 0.55), int(self.height * 0.55)
        cv2.rectangle(ai_frame, (mask_x1, mask_y1), (self.width, self.height), (0, 0, 0), -1)

        # 3. PREDICT (Non-streamed to avoid Error 121 crashes)
        results = self.model.predict(
            source=ai_frame, 
            conf=0.32, 
            imgsz=640, 
            device='mps', 
            verbose=False
        )
        
        best_target = None
        min_dist = float('inf')

        if results and len(results[0].boxes) > 0:
            # Move data to CPU immediately to fix Error 121
            boxes = results[0].boxes.cpu() 
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                w, h = x2 - x1, y2 - y1
                
                # Human Aspect Ratio Filter
                if (h / w) < 1.3 or h < 25: continue

                # Snap to Upper Chest/Neck
                obj_x = int((x1 + x2) / 2)
                obj_y = int(y1 + h * 0.12) 

                dist = np.sqrt((obj_x - self.center_x)**2 + (obj_y - self.center_y)**2)
                
                if dist <= 180 and dist < min_dist:
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
            # Decay back to center
            self.smooth_x = (self.center_x * 0.1) + (self.smooth_x * 0.9)
            self.smooth_y = (self.center_y * 0.1) + (self.smooth_y * 0.9)

        # Rel x/y for Titan Two
        rel_x, rel_y = int(self.smooth_x - self.center_x), int(self.smooth_y - self.center_y)
        gcvdata = bytearray([1])
        gcvdata.extend(rel_x.to_bytes(2, byteorder='big', signed=True))
        gcvdata.extend(rel_y.to_bytes(2, byteorder='big', signed=True))
        
        return (frame, gcvdata)

    def __del__(self):
        pass
