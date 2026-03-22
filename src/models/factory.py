from src.models.blip_classifier import BlipClassifier
from src.models.parallel_cnn import ParallelCNN


def build_model(cfg):
    model_name = cfg["model"]["name"]

    if model_name == "blip_baseline":
        return BlipClassifier(cfg)

    if model_name == "parallel_cnn":
        return ParallelCNN(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            branch_kernels=cfg["model"]["branch_kernels"],
            base_channels=cfg["model"]["base_channels"],
            dropout=cfg["model"]["dropout"],
        )

    raise ValueError(f"Unknown model name: {model_name}")