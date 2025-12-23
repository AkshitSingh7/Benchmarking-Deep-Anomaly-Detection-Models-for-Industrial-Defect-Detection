from src.dataset import get_datamodule
from src.model import get_model, get_engine
from src.visualize import visualize_inference
import torch

def main():
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not found. Training will be slow.")

    print("🚀 Starting Industrial Defect Detection Pipeline...")

    # 1. Setup Data
    print("\n[1/4] Loading & Verifying Data...")
    # You can change category to 'hazelnut', 'transistor', etc.
    datamodule = get_datamodule(category="bottle")
    
    # 2. Setup Model
    print("\n[2/4] Initializing PatchCore Model...")
    model = get_model()
    engine = get_engine()
    
    # 3. Train & Test
    print("\n[3/4] Training & Benchmarking...")
    # Fit (Train)
    engine.fit(datamodule=datamodule, model=model)
    print("✅ Training Complete.")
    
    # Test
    test_results = engine.test(datamodule=datamodule, model=model)
    score = test_results[0]['image_AUROC']
    print(f"\n🏆 FINAL SCORE (AUROC): {score:.4f}")
    
    # 4. Visualize
    print("\n[4/4] Generating Visual Forensics...")
    # Generates the heatmap for the first image in the test set
    visualize_inference(model, datamodule, index=0, save_path="result_heatmap.png")
    
    print("\n✅ Pipeline Finished. Check 'result_heatmap.png' for the heatmap.")

if __name__ == "__main__":
    main()
