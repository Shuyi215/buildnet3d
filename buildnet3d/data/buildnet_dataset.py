from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path


class BuildNetDataset(InputDataset):
    """Dataset that returns RGB images along with semantic labels and foreground masks."""

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + [
        "mask",
        "semantics",
    ]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        
        if not isinstance(self.metadata.get("semantics"), Semantics):
            raise ValueError("Missing or invalid 'semantics' metadata in dataparser_outputs.")
        
        self.semantics = self.metadata["semantics"]
        self.mask_indices = torch.tensor([
            self.semantics.classes.index(mask_class)
            for mask_class in self.semantics.mask_classes
        ]).view(1, 1, -1)

        self.camera_to_worlds = self.metadata.get("camera_to_worlds")
        self.transform = self.metadata.get("transform")

    def get_metadata(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # handle segmentation and its mask
        filepath = self.semantics.filenames[data["image_idx"]]
        semantic_label, mask = get_semantics_and_mask_tensors_from_path(
            filepath=filepath,
            mask_indices=self.mask_indices,
            scale_factor=self.scale_factor,
        )
        mask = mask.type(torch.float32)
        # H, W = semantic_label.shape[:2]
        
        # # Inputdataset overlooked alpha channel. load again to get RGBA data
        # raw_image = self.get_numpy_image(data["image_idx"]) 
    
        # if raw_image.shape[-1] == 4:
        #     # extract Alpha as Mask (0 for background, 1 for foreground)
        #     alpha_mask = torch.from_numpy(raw_image[..., 3:4].astype("float32") / 255.0)
        #     # binarize alpha mask
        #     mask *= alpha_mask   # merge alpha with semantic mask
        
        # if "mask" in data:
        #     mask *= data["mask"].type(torch.float32)   # merge with mask from dataparser

        return {"semantics": semantic_label, 
                "mask": mask,
                "fg_mask": mask,
                } 
