from src.dataset import get_datamodule
from src.model import get_model, get_engine
from src.visualize import visualize_inference
import torch

def run_benchmark(model_name, datamodule, epochs=1):
    """
    Helper to train and test a specific model.
    """
    print(f"\n--- 🤖 Benchmarking: {model_name.upper()} ---")
    
    # 1. Init
    model = get_model(model_name)
    engine = get_engine()
    
    # Ganomaly needs more epochs to learn anything at all
    if model_name == "ganomaly":
        engine.trainer.max_epochs = epochs 
    
    # 2. Train
    print(f"   Training for {engine.trainer.max_epochs} epoch(s)...")
    engine.fit(datamodule=datamodule, model=model)
    
    # 3. Test
    print("   Testing...")
    test_results = engine.test(datamodule=datamodule, model=model)
    score = test_results[0]['image_AUROC']
    
    print(f"   📉 {model_name.upper()} Result (AUROC): {score:.4f}")
    return model, score

def main():
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not found.")

    print("🚀 Starting Industrial Benchmark Pipeline...")

    # 1. Load Data once
    print("\n[1/3] Loading Data...")
    datamodule = get_datamodule(category="bottle")

    # 2. Run Ganomaly (The "Bad" Baseline)
    # We give it 5 epochs just to try, but it usually fails on complex textures
    gan_model, gan_score = run_benchmark("ganomaly", datamodule, epochs=5)

    # 3. Run PatchCore (The "SOTA" Solution)
    patch_model, patch_score = run_benchmark("patchcore", datamodule, epochs=1)

    # 4. Final Verdict
    print("\n" + "="*40)
    print(f"🏆 FINAL VERDICT")
    print(f"   Ganomaly:  {gan_score:.4f}")
    print(f"   PatchCore: {patch_score:.4f}")
    print("="*40)

    # 5. Visualize the Winner (PatchCore)
    print("\n[Visual Forensics] Generating Heatmap for the winner...")
    visualize_inference(patch_model, datamodule, index=0, save_path="result_heatmap.png")
    print("✅ Benchmark Complete.")

if __name__ == "__main__":
    main()
