from raft_lib.raft import RAFT as _RAFT
import os
from typing import Literal
from pathlib import Path
import requests
import torch
from argparse import Namespace
from tqdm import tqdm
import tempfile
import shutil
from zipfile import ZipFile

__all__ = ["RAFT"]


class RAFTWrapper(_RAFT):
    """
    Wrapper for the original RAFT model to ease the loading
    of the pretrained models
    """

    def __init__(
        self,
        dataset: Literal[None, "kitti", "sintel", "chairs", "small", "things"] = None,
        mixed_precision: bool = False,
    ):
        super().__init__(
            Namespace(
                small=dataset == "small",
                mixed_precision=mixed_precision,
                alternate_corr=False,
            )
        )
        if dataset is not None:
            models_root = Path(torch.hub.get_dir()) / "raft_models"
            if not models_root.exists():
                models_root.mkdir(exist_ok=True, parents=True)
                torch.hub.download_url_to_file(
                    "https://www.dropbox.com/s/raw/4j4z58wuv8o0mfz/models.zip",
                    models_root / "models.zip",
                )
                with open(models_root / "models.zip", "rb") as f:
                    ZipFile(f).extractall(models_root)
                os.remove(models_root / "models.zip")

            path = models_root / f"models/raft-{dataset}.pth"
            self.load_state_dict(
                {".".join(k.split(".")[1:]): v for k, v in torch.load(path).items()}
            )
