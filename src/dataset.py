"""Dataset module for loading and preprocessing retinal fundus images."""

import os
from typing import Callable, Dict, Optional, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class RetinalDataset(Dataset):
    """Custom PyTorch Dataset for Retinal Fundus Image Classification.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'file_path' and 'label' columns.
        label_map (Dict[str, int]): Dictionary mapping string labels to integer IDs.
        transform (callable, optional): Optional torchvision transforms to apply to images.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_map: Dict[str, int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Fetches the image and its corresponding label at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple containing the processed image tensor and integer label.
        """
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        label_str = row["label"]

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image at path {file_path}: {e}")

        if label_str not in self.label_map:
            raise ValueError(f"Label '{label_str}' not found in the provided label_map.")
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label
