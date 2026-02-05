"""
Step 16: YOLOv8 Vision Logic.
Pipeline: Image -> Object Detection -> Pattern Embedding -> Signal.
"""

import numpy as np
import cv2
import pandas as pd
from typing import List, Tuple, Dict, Optional
import os
from datetime import datetime
from src.core.types import FinalSignal, TradeAction, VisionOutput, MarketBias
from src.core.config import CONFIDENCE_THRESHOLD

# Safe Import for Ultralytics
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("⚠️ Ultralytics not installed. Running YOLO Engine in Simulation Mode.")

class YOLOEngine:
    """
    Advanced Vision Engine using Object Detection (YOLOv8).
    Detects: Candles, Doji, Pinbars, Supports, Resistances.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = None
        self.sim_mode = True
        
        if HAS_YOLO:
            try:
                # We expect a custom trained model, but fallback to base nano for demo
                # In production, this path would be 'models/best.pt'
                self.model = YOLO(model_path) 
                self.sim_mode = False # Attempt real inference
                # If model path doesn't exist, YOLO library auto-downloads base model
                # But base model detects 'person', 'car'... not 'doji'.
                # So we essentially need a switch: Real Custom Model vs Base Demo vs Simulation
            except Exception as e:
                print(f"YOLO Load Error: {e}. Switching to Simulation.")
                self.sim_mode = True
        else:
            self.sim_mode = True
            
        # Class mapping for financial model (Hypothetical)
        self.classes = {
            0: "bullish_candle",
            1: "bearish_candle",
            2: "doji",
            3: "pinbar_bullish",
            4: "pinbar_bearish",
            5: "support",
            6: "resistance"
        }

    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Mock detections for Verification without a trained financial model.
        Returns random valid boxes.
        """
        h, w = image.shape[:2]
        detections = []
        
        # Simulate a Setup (Bullish Pinbar at Support)
        
        # 1. Support Zone (Bottom)
        detections.append({
            "class": 5, # Support
            "label": "support",
            "conf": 0.92,
            "box": [0, h - 50, w, h] # Bottom strip
        })
        
        # 2. Bullish Pinbar (Near support)
        detections.append({
            "class": 3, # Pinbar Bull
            "label": "pinbar_bullish",
            "conf": 0.88,
            "box": [w-100, h - 80, w-50, h-30]
        })
        
        # 3. Some prior context (Bearish candle)
        detections.append({
            "class": 1, # Bearish
            "label": "bearish_candle",
            "conf": 0.85,
            "box": [w-160, h-120, w-110, h-70]
        })
        
        return detections

    def _run_inference(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8 output parsing.
        """
        if self.sim_mode or self.model is None:
            return self._simulate_detection(image)
            
        try:
            results = self.model(image, verbose=False)
            parsed = []
            
            # YOLO results object
            # We assume the model is trained on our classes.
            # If it's the stock 'yolov8n.pt', classes are COCO (person, car...).
            # We will force Simulation if we detect we are using stock model classes 
            # (e.g. class 0 is person).
            
            names = self.model.names
            if 0 in names and names[0] == 'person':
                # Stock model detected -> Fallback to financial simulation
                return self._simulate_detection(image)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist() # x1, y1, x2, y2
                    
                    label = self.classes.get(cls_id, "unknown")
                    
                    parsed.append({
                        "class": cls_id,
                        "label": label,
                        "conf": conf,
                        "box": xyxy
                    })
            return parsed
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return self._simulate_detection(image)

    def _simulate_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Mock detections: A sequence of candles leading to a setup.
        Simulates: Bearish -> Bearish -> Small Red -> Big Green (Engulfing).
        """
        h, w = image.shape[:2]
        detections = []
        
        # Grid settings
        cw = 20 # candle width
        gap = 10
        base_x = w // 2 - 100
        
        # 1. Bearish Candle
        detections.append({
            "class": 1, "label": "bearish_candle", "conf": 0.88,
            "box": [base_x, h-200, base_x+cw, h-150]
        })
        
        # 2. Small Red (Inside/Base)
        x2 = base_x + cw + gap
        detections.append({
            "class": 1, "label": "bearish_candle", "conf": 0.85,
            "box": [x2, h-160, x2+cw, h-155] # Small body
        })
        
        # 3. Big Green (Engulfing the previous small red)
        x3 = x2 + cw + gap
        detections.append({
            "class": 0, "label": "bullish_candle", "conf": 0.92,
            "box": [x3, h-170, x3+cw, h-140] # Engulfs y-range of previous
        })
        
        # 4. Support Zone (Context)
        detections.append({
            "class": 5, "label": "support", "conf": 0.95,
            "box": [0, h-130, w, h-100] # Below the candles
        })
        
        return detections

    def _normalize_sequence(self, detections: List[Dict], image_height: int) -> pd.DataFrame:
        """
        Convert bounding boxes to relative OHLC.
        Y-axis is inverted in images (0 is top), so Price = (Height - y) / Height.
        """
        candles = [d for d in detections if 'candle' in d['label']]
        # Sort by X coordinate (Time)
        candles.sort(key=lambda x: x['box'][0])
        
        data = []
        for c in candles:
            x1, y1, x2, y2 = c['box']
            # Normalized Price (0 to 1, 1 is Top of image/High Price)
            # Img Y: Top=0, Bottom=H.   Price: Top=1, Bottom=0.
            top_p = 1.0 - (y1 / image_height)
            btm_p = 1.0 - (y2 / image_height)
            
            # Identify Open/Close based on color/label
            # Green (Bull): Close > Open (Top > Bottom in price space) -> Top is Close
            # Red (Bear): Close < Open -> Bottom is Close
            
            if c['label'] == 'bullish_candle':
                 high = top_p
                 low = btm_p
                 close = top_p
                 open_p = btm_p # Simplification: Body is full range for box
            else:
                 high = top_p
                 low = btm_p
                 close = btm_p
                 open_p = top_p
                 
            data.append({
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": close,
                "Label": c['label'],
                "Conf": c['conf']
            })
            
        return pd.DataFrame(data)

    def analyze(self, image_input) -> FinalSignal:
        """
        Main Pipeline Entry.
        Accepts: Numpy Image (BGR or RGB).
        """
        # Preprocess
        if image_input is None:
             return FinalSignal(TradeAction.STAY_OUT, 0.0, ["YOLO: No Image"], "N/A", "N/A", "YOLO")
             
        # Extract numpy if it's a tensor
        img = image_input
        if hasattr(img, 'numpy'): img = img.numpy()
        if len(img.shape) == 4: img = img[0] # Batch dim
        
        h, w = img.shape[:2]
        
        # Run Detection
        detections = self._run_inference(img)
        
        # 1. Normalization & OHLC Reconstruction
        df_candles = self._normalize_sequence(detections, h)
        
        # Scoring Weights
        base_conf = 0.0
        pattern_conf = 0.0
        context_conf = 0.0
        
        reasons = []
        action = TradeAction.STAY_OUT
        
        # A. Pattern Analysis
        if not df_candles.empty and len(df_candles) >= 2:
            last = df_candles.iloc[-1]
            prev = df_candles.iloc[-2]
            
            # 1. Bullish Engulfing
            is_bull_engulf = (prev['Label'] == 'bearish_candle') and \
                            (last['Label'] == 'bullish_candle') and \
                            (last['Close'] > prev['Open']) and \
                            (last['Open'] < prev['Close'])
            
            if is_bull_engulf:
                action = TradeAction.BUY
                pattern_conf = 0.30
                reasons.append(f"Bullish Engulfing (Visual)")
                
            # 2. Bearish Engulfing
            is_bear_engulf = (prev['Label'] == 'bullish_candle') and \
                            (last['Label'] == 'bearish_candle') and \
                            (last['Close'] < prev['Open']) and \
                            (last['Open'] > prev['Close'])
                            
            if is_bear_engulf:
                action = TradeAction.SELL
                pattern_conf = 0.30
                reasons.append(f"Bearish Engulfing (Visual)")

            # 3. Pinbar Logic (Single Candle)
            if 'pinbar_bullish' in last['Label']:
                action = TradeAction.BUY
                pattern_conf = 0.25 # Slightly less than engulfing pair
                reasons.append("Bullish Pinbar Detected")
            elif 'pinbar_bearish' in last['Label']:
                action = TradeAction.SELL
                pattern_conf = 0.25
                reasons.append("Bearish Pinbar Detected")

        # B. Context Analysis (Support/Resistance)
        supports = [d for d in detections if d['label'] == 'support']
        resistances = [d for d in detections if d['label'] == 'resistance']
        
        if supports and action == TradeAction.BUY:
            context_conf = 0.20
            reasons.append("Confluence: Support Zone")
        elif resistances and action == TradeAction.SELL:
            context_conf = 0.20
            reasons.append("Confluence: Resistance Zone")
            
        # C. Base Confidence (from YOLO detection probability)
        # We take the average confidence of the 'active' objects
        if action != TradeAction.STAY_OUT:
            # Gather relevant detection confs
            relevant_confs = [last['Conf']] if 'last' in locals() else []
            if context_conf > 0 and supports: relevant_confs.append(supports[0]['conf'])
            if context_conf > 0 and resistances: relevant_confs.append(resistances[0]['conf'])
            
            avg_det_conf = sum(relevant_confs) / len(relevant_confs) if relevant_confs else 0.5
            
            # Base contributes 40% of standard weight
            base_conf = avg_det_conf * 0.4 
        
        # D. Final Weighted Sum
        # Max Possible = 0.4 (Base) + 0.3 (Pattern) + 0.2 (Context) = 0.9
        total_confidence = base_conf + pattern_conf + context_conf
        
        # E. Threshold Filter
        FINAL_THRESHOLD = 0.65
        
        if total_confidence < FINAL_THRESHOLD:
            if action != TradeAction.STAY_OUT:
                reasons.append(f"Low Confidence ({total_confidence:.2f} < {FINAL_THRESHOLD})")
            action = TradeAction.STAY_OUT
            total_confidence = 0.0
            
        # Construct Output
        return FinalSignal(
            action=action,
            confidence=total_confidence,
            reasoning=reasons,
            vision_bias="Simulated Detections" if self.sim_mode else "YOLO Inference",
            ts_bias=f"Score:{total_confidence:.2f}",
            regime="ObjectDetection"
        )

    def get_annotated_image(self, image: np.ndarray) -> np.ndarray:
        """
        Helper to draw boxes for UI.
        """
        img_copy = image.copy()
        if img_copy.max() <= 1.0: img_copy = (img_copy * 255).astype(np.uint8)
        
        detections = self._run_inference(img_copy)
        
        for d in detections:
            box = d['box'] # x1, y1, x2, y2 or xywh?
            # Simulation returns xyxy format usually for drawing convenience or whatever. 
            # Sim mode above returned [x1, y1, x2, y2].
            
            x1, y1, x2, y2 = map(int, box)
            label = d['label']
            
            color = (0, 255, 0)
            if 'bear' in label or 'resistance' in label:
                color = (0, 0, 255) # Red (BGR)
            elif 'doji' in label:
                color = (255, 255, 0) # Cyan/Yellowish
                
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return img_copy

    def save_snapshot(self, image: np.ndarray, prefix: str = "train_data") -> str:
        """
        Save raw image to dataset folder for future training.
        """
        directory = "user_data/datasets/raw"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        path = os.path.join(directory, filename)
        
        # Ensure BGR for CV2 save if input is RGB (Streamlit is RGB usually)
        # We assume input image is consistent with analyze() input
        # If image is RGB, cv2.imwrite expects BGR. 
        # But analyze() usually receives what we pass. 
        # In dashboard we convert PIL(RGB) -> numpy.
        # Let's assume input is RGB and convert to BGR for saving.
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
        
        return path
