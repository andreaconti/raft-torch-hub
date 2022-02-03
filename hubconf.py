"""
Configuration file for Torch Hub
"""
dependencies = [
    "torch",
    "requests",
    "tqdm",
]

import typing as _t


def raft(
    pretrained: bool = False,
    dataset: _t.Literal["kitti", "sintel", "chairs", "small", "things"] = "sintel",
):
    """
    RAFT model as in https://github.com/princeton-vl/RAFT, with the
    same interface.

    Parameters
    ----------
    pretrained: bool
        if true loads the official trained models from https://github.com/princeton-vl/RAFT
    dataset: kitti | sintel | chairs | small | things
        the dataset on which the pretrained model has been trained, default sintel

    Returns
    -------
    out: nn.Model
        the RAFT model

    Examples
    --------
    To retrieve the optical flow

    >>> model = raft(pretrained=True, dataset="kitti")
    >>> flow, flow_up = model(img_1, img_2, n_iters=12, flow_init=None, test_mode=True)
    """
    from lib.wrapper import RAFT

    model = RAFT(dataset if pretrained else None)
    return model
