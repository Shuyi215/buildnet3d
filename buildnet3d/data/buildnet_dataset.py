from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import (
    get_image_mask_tensor_from_path,
    get_semantics_and_mask_tensors_from_path,
)


class BuildNetDataset(InputDataset):
    """RGB images, optional binary foreground masks, optional semantic labels.

    Foreground masks are loaded from ``metadata["fg_mask_filenames"]`` (0/255 binary
    PNGs) and are independent of semantic segmentation.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
        "mask",
        "semantics",
        "fg_mask",
    ]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        self.semantics: Optional[Semantics] = None
        if isinstance(self.metadata.get("semantics"), Semantics):
            self.semantics = self.metadata["semantics"]

        self.fg_mask_filenames: Optional[List[Path]] = self.metadata.get("fg_mask_filenames")

        self.camera_to_worlds = self.metadata.get("camera_to_worlds")
        self.transform = self.metadata.get("transform")

    def get_metadata(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        metadata: Dict[str, torch.Tensor] = {}
        image_idx = data["image_idx"]

        if self.fg_mask_filenames is not None:
            # 255 -> True (foreground), 0 -> False (background)
            fg_mask = get_image_mask_tensor_from_path(
                filepath=self.fg_mask_filenames[image_idx],
                scale_factor=self.scale_factor,
            )
            metadata["fg_mask"] = fg_mask.type(torch.float32)

        if self.semantics is not None:
            filepath = self.semantics.filenames[image_idx]
            # Empty mask_indices: load labels only; do not derive fg_mask from semantics.
            semantic_label, _ = get_semantics_and_mask_tensors_from_path(
                filepath=filepath,
                mask_indices=[],
                scale_factor=self.scale_factor,
            )
            metadata["semantics"] = semantic_label

        return metadata
