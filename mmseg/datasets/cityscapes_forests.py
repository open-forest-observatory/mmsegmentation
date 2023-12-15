# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesForestsDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    METAINFO = dict(
        classes=(
            "ABCO",
            "ABMA",
            "CADE",
            "PI",
            "PICO",
            "PIJE",
            "PILA",
            "PIPO",
            "SALSCO",
            "TSME",
        ),
        palette=[
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
        ],
    )

    def __init__(
        self,
        img_suffix="_rgb.JPG",
        seg_map_suffix="_segmentation.png",
        classes=None,
        **kwargs
    ) -> None:
        if classes is not None:
            self.METAINFO["classes"] = classes
            self.METAINFO["palette"] = self.METAINFO["palette"][: len(classes)]

        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
