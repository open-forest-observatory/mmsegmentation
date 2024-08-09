# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.utils.class_names import cityscapes_arbitrary_classes_classes, cityscapes_arbitrary_classes_palette
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CityscapesArbitraryClassesDataset(BaseSegDataset):
    """Cityscapes dataset.

    # TODO update comment
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    METAINFO = dict(
        classes=cityscapes_arbitrary_classes_classes(n_classes=10),
        palette=None,
    )

    def __init__(
        self,
        img_suffix="_rgb.JPG",
        seg_map_suffix="_segmentation.png",
        classes=None,
        **kwargs
    ) -> None:
        # Set the classes if provided
        if classes is not None:
            self.METAINFO["classes"] = classes

        n_classes = len(self.METAINFO["classes"])
        # Define the palette on the fly using matplotlib colormaps
        palette = cityscapes_arbitrary_classes_palette(n_classes=n_classes)
        # Set the palette
        self.METAINFO["palette"] = palette

        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
