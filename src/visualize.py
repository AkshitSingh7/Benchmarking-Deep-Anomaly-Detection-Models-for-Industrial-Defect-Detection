import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_inference(model, datamodule, index=0, save_path=None):
    """
    Runs inference on a single test image and plots the anomaly heatmap.
    """
    # 1. Get Data
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    
    if index >= len(batch["image"]):
        print(f"⚠️ Index {index} is out of bounds for this batch.")
        return

    # 2. Run Inference
    img_tensor = batch["image"][index].unsqueeze(0).to("cuda")
    model.eval()
    model.to("cuda")
    
    with torch.no_grad():
        predictions = model(img_tensor)

    # 3. Smart Extraction (Handle API variations)
    anomaly_map = None
    
    # Helper: checks if a tensor looks like an image map (Height, Width)
    def is_map(tensor):
        return hasattr(tensor, "ndim") and tensor.ndim >= 2 and tensor.shape[-1] > 1

    if isinstance(predictions, tuple):
        # If output is (score, map) or (map, score), find the map
        for item in predictions:
            if is_map(item):
                anomaly_map = item
                break
    elif isinstance(predictions, dict):
        anomaly_map = predictions.get("anomaly_maps")
    elif is_map(predictions):
        anomaly_map = predictions

    if anomaly_map is None:
        print("❌ Error: Could not extract anomaly map from model output.")
        return

    # 4. Process for Plotting
    anomaly_map = anomaly_map.cpu().numpy().squeeze()
    
    # Un-normalize original image
    original_img = batch["image"][index].cpu().numpy().transpose(1, 2, 0)
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

    # 5. Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original Image #{index}")
    axes[0].axis("off")

    axes[1].imshow(anomaly_map, cmap="inferno")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(original_img)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved visualization to {save_path}")
    
    plt.show()
