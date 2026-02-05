import numpy as np
import cv2
from src.engines.yolo_logic import YOLOEngine
from src.core.types import TradeAction

def test_yolo_pipeline():
    print("🚀 [TEST] Phase 20: YOLOv8 Vision Pipeline")
    
    # 1. Init Engine
    # Expect Simulation Mode warning if Ultralytics missing/no model
    engine = YOLOEngine()
    print(f"Engine Mode: {'Simulated' if engine.sim_mode else 'Real Inference'}")
    
    # 2. Mock Image (Black Canvas)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 3. Run Analysis
    print("Running Analysis...")
    signal = engine.analyze(dummy_img)
    
    print(f"Action: {signal.action}")
    print(f"Confidence: {signal.confidence}")
    print(f"Reasons: {signal.reasoning}")
    
    # 4. Assertions (Based on Simulation Logic)
    # Sim logic hardcodes a Bullish Pinbar setup -> BUY
    if engine.sim_mode:
        assert signal.action == TradeAction.BUY, f"Expected BUY in sim, got {signal.action}"
        assert signal.confidence > 0.8, "Expected high confidence for sim setup"
        print("✅ Signal Logic Correct")
        
    # 5. Visual Check
    print("Generating Annotated Image...")
    annotated = engine.get_annotated_image(dummy_img)
    assert annotated.shape == dummy_img.shape, "Annotated image shape mismatch"
    print("✅ Annotation Generated")
    
    print("✅ Phase 20 Tests Passed!")

if __name__ == "__main__":
    test_yolo_pipeline()
