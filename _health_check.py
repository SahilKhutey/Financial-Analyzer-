import sys
import os
import traceback

# Add current directory to path
sys.path.append(os.getcwd())

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ Import successful: {module_name}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {module_name}")
        print(traceback.format_exc())
        return False

def check_class(module_name, class_name):
    try:
        module = sys.modules[module_name]
        if hasattr(module, class_name):
            print(f"✅ Class found: {class_name} in {module_name}")
            return True
        else:
            print(f"❌ Class not found: {class_name} in {module_name}")
            return False
    except Exception:
        print(f"❌ Could not check class {class_name} in {module_name} due to previous errors")
        return False

print("Starting Health Check...")

modules_to_check = [
    "src.core.config",
    "src.core.gates",
    "src.core.risk",
    "src.core.types",
    "src.data.capture",
    "src.data.ingestion",
    "src.data.processing",
    "src.engines.smart_money",
    "src.engines.math_engine",
    "src.engines.deep_analytic",
    "src.engines.ml_production",
    "src.engines.vision",
    "src.engines.backtest",
    "src.ui.dashboard"
]

all_passed = True
for mod in modules_to_check:
    if not check_import(mod):
        all_passed = False

if all_passed:
    # Check specific classes
    check_class("src.engines.smart_money", "SmartMoneyEngine")
    check_class("src.engines.math_engine", "MathPredictionEngine")
    check_class("src.engines.deep_analytic", "DeepAnalyticalEngine")
    check_class("src.engines.ml_production", "MLProductionEngine")
    
    # Test load_engines from dashboard
    print("Testing dashboard load_engines()...")
    try:
        from src.ui.dashboard import load_engines
        engines = load_engines()
        print(f"✅ load_engines() returned {len(engines)} items")
        if len(engines) == 14:
            print("✅ Engine count matches unpacking expectation (14)")
        else:
            print(f"❌ Engine count mismatch! Expected 14, got {len(engines)}")
        
        # Verify unpacking consistency manually checking expected count in dashboard source is hard via import
        # but we can check if it matches the variables we know are unpacked
        # The bug is likely in the number of items vs unpack variables. 
        # But we can't easily execute main() without streamlit context entirely.
        # However, checking if it runs without error is a good step.
    except Exception as e:
        print(f"❌ load_engines() failed: {e}")
        traceback.print_exc()

print("Health Check Complete.")
