from anomalib.models import Patchcore
from anomalib.engine import Engine
import warnings

def get_model(backbone="resnet18"):
    """
    Returns the configured PatchCore model.
    Using ResNet18 + 1% coreset sampling for optimal Colab/T4 performance.
    """
    model = Patchcore(
        backbone=backbone,
        coreset_sampling_ratio=0.01,
    )
    return model

def get_engine():
    """
    Returns the Training Engine (Trainer).
    CRITICAL: Disables progress bar to prevent RecursionError in some environments.
    """
    # Filter warnings to keep output clean
    warnings.filterwarnings("ignore")
    
    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=False  # <--- Fixes the crashing/recursion issue
    )
    return engine
