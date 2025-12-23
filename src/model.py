from anomalib.models import Patchcore, Ganomaly
from anomalib.engine import Engine
import warnings

def get_model(model_name="patchcore"):
    """
    Factory function to return the requested model architecture.
    """
    if model_name.lower() == "patchcore":
        # SOTA Performance (AUROC ~1.0)
        return Patchcore(
            backbone="resnet18",
            coreset_sampling_ratio=0.01,
        )
    elif model_name.lower() == "ganomaly":
        # Generative Baseline (AUROC ~0.50 on simple setup)
        return Ganomaly()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_engine():
    """
    Returns the Training Engine (Trainer).
    CRITICAL: Disables progress bar to prevent RecursionError.
    """
    warnings.filterwarnings("ignore")
    
    # Note: max_epochs is 1 for PatchCore, but Ganomaly typically needs more (e.g., 20-50).
    # We set a default here, but you can override it in main.py if needed.
    engine = Engine(
        max_epochs=1, 
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=False 
    )
    return engine
