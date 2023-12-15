from pathlib import Path

import numpy as np
import argparse

# from mmseg.apis import init_model, inference_model
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--input-tile", type=Path)
    parser.add_argument("--output-tile", type=Path)
    parser.add_argument("--brightness-scale", type=float)
    parser.add_argument("--crop-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)

    args = parser.parse_args()
    return args


if __name__:
    args = parse_args()

    args.output_tile.parent.mkdir(exist_ok=True)

    # model = init_model(str(args.config_path), str(args.checkpoint_path))

    # TODO do something with this to make sure we only get the one file
    class MyRasterDataset(RasterDataset):
        filename_regex = args.input_tile.name

    dataset = MyRasterDataset(str(args.input_tile.parent))
    breakpoint()

    sampler = GridGeoSampler(
        dataset,
        size=(args.crop_size, args.crop_size),
        stride=(args.stride, args.stride),
    )
    dataloader = DataLoader(
        dataset, sampler=sampler, collate_fn=stack_samples, num_workers=1
    )

    # We need to create a default no-data tiff of the same size
    # This should have the same resolution, crs, and pixel count as the input, but only one band

    for i, batch in enumerate(dataloader):
        img = batch["image"]
        img = img * args.brightness_scale

        # TODO see if dimensions or anything need to be changed
        # result = inference_model(model, img)
        # seg = result.pred_sem_seg.data.cpu().numpy()[0].astype(np.uint8)

        # Write out the data to the correct place in the output tile
        ## Find the pixel coords of the bounding box in the output image
        ## Do a windowed write to this location

        print(i)
