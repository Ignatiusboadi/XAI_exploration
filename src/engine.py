# Training and validation engine module for retinal fundus classification.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Tuple


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Trains the model for a single epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Optimization algorithm (e.g., Adam, AdamW).
        device (torch.device): Device to run computations on (cuda/cpu).

    Returns:
        Tuple[float, float]: Average training loss and training accuracy for the epoch.
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Evaluates the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computations on (cuda/cpu).

    Returns:
        Tuple[float, float]: Average validation loss and validation accuracy.
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc
