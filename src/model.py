# Model architecture module for retinal fundus classification.

import torch
import torch.nn as nn
import torchvision.models as models


def create_retinal_model(model_name: str = "resnet50", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Factory function to initialize a pre-trained model and adapt its classification head.

    Args:
        model_name (str): Name of the architecture ('resnet50', 'efficientnet_b0', etc.).
        num_classes (int): Number of target output classes (7 for this project).
        pretrained (bool): Whether to load ImageNet pre-trained weights.

    Returns:
        nn.Module: The configured PyTorch model ready for training.
    """
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model architecture '{model_name}' is not supported.")

    return model