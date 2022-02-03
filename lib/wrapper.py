from lib.raft import RAFT as _RAFT
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


def download_file(url, save_path, chunk_size=1024, verbose=True):
    """
    Downloads a zip file from an `url` into a zip file in the
    provided `save_path`.
    """
    r = requests.get(url, stream=True)
    zip_name = url.split("/")[-1]

    if "Content-Length" in r.headers:
        content_length = int(r.headers["Content-Length"]) / 10**6
    else:
        content_length = None

    if verbose:
        bar = tqdm(total=content_length, unit="Mb", desc="download " + zip_name)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            if verbose:
                bar.update(chunk_size / 10**6)

    if verbose:
        bar.close()


def _download_models():
    models_path = Path(__file__).parent / "models"
    if not models_path.exists():
        tmpdir = Path(tempfile.mkdtemp())
        download_file(
            "https://www.dropbox.com/s/raw/4j4z58wuv8o0mfz/models.zip",
            tmpdir / "models.zip",
        )
        with open(tmpdir / "models.zip", "rb") as f:
            ZipFile(f).extractall(Path(__file__).parent)
        shutil.rmtree(tmpdir, ignore_errors=True)


class RAFT(_RAFT):
    """
    Wrapper for the original RAFT model to ease the loading
    of the pretrained models
    """

    def __init__(
        self,
        dataset: Literal[None, "kitti", "sintel", "chairs", "small", "things"] = None,
    ):
        super().__init__(
            Namespace(
                small=dataset == "small", mixed_precision=False, alternate_corr=False
            )
        )
        if dataset is not None:
            _download_models()
            path = Path(__file__).parent / f"models/raft-{dataset}.pth"
            self.load_state_dict(
                {".".join(k.split(".")[1:]): v for k, v in torch.load(path).items()}
            )
