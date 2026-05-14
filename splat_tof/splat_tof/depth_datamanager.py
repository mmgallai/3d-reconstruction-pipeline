"""FullImageDatamanager that uses DepthDataset so depth maps are in the batch.

nerfstudio's stock FullImageDatamanager hardcodes InputDataset, which only
loads RGB. For depth-supervised training the batch needs `depth_image`. We
subclass to swap the dataset class and re-export with a new config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.depth_dataset import DepthDataset


@dataclass
class FullImageDepthDatamanagerConfig(FullImageDatamanagerConfig):
    """FullImage datamanager that loads sensor depths from `depth_file_path`."""
    _target: Type = field(default_factory=lambda: FullImageDepthDatamanager)


class FullImageDepthDatamanager(FullImageDatamanager):
    """FullImageDatamanager but dataset_type=DepthDataset (loads depth_image)."""
    config: FullImageDepthDatamanagerConfig
    dataset_type = DepthDataset
