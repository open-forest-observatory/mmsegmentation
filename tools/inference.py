from mmseg.apis import init_model, inference_model
from pathlib import Path
from imageio import imwrite, imread
import numpy as np
import argparse
import imagesize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--store-probs", action="store_true")
    parser.add_argument("--extension", default="")

    args = parser.parse_args()
    return args


def save_result(result, output_file, store_probs):
    # Create the the folder for the file
    Path(output_file.parent).mkdir(exist_ok=True, parents=True)
    if store_probs:
        output_file = output_file.with_suffix(".npy")
        seg_logits = result.seg_logits.data.cpu().numpy()
        seg_logits = np.transpose(seg_logits, (1, 2, 0))
        np.save(output_file, seg_logits)
    else:
        seg = result.pred_sem_seg.data.cpu().numpy()[0].astype(np.uint8)
        output_file = output_file.with_suffix(".png")
        imwrite(output_file, seg)

def get_image_shape(file):
    """Return the (h, w) tuple of image shape if it's a image, otherwise None"""
    try:
        shape = imagesize.get(file)
        if shape == (-1, -1):
            return None
        return shape
    except:
        return None

if __name__ == "__main__":

    args = parse_args()

    # Get all files
    all_files = list(args.image_folder.rglob("*" + args.extension))
    # Get the shape of all images. Will be None if not an image
    image_shapes = []
    for file in all_files:
        image_shapes.append(get_image_shape(file))
    # Merge the shapes with the paths
    shape_file_list = zip(image_shapes, all_files)
    # Filter out the tuples that don't correspond to an image
    shape_file_list = list(filter(lambda x: x[0] is not None, shape_file_list))
    unique_shapes = np.unique([shape_file[0] for shape_file in shape_file_list], axis=0)
    # Convert back into tuples
    unique_shapes = [tuple(unique_shape) for unique_shape in unique_shapes]

    model = init_model(str(args.config_path), str(args.checkpoint_path))

    print(f"Unique image shapes are {unique_shapes}")

    for unique_shape in unique_shapes:
        print(f"Processing images with {unique_shape} shape")
        # Extract the filenames corresponding to images of that shape
        matching_files = [x[1] for x in filter(lambda x: x[0] == unique_shape, shape_file_list)]

        n_files = len(matching_files)
        for i in range(0, len(matching_files), args.batch_size):
            print(f"index: {i}/{n_files}", end="\r")
            batch_files = matching_files[i : i + args.batch_size]

            results = inference_model(model, [str(x) for x in batch_files])

            rel_paths = [x.relative_to(args.image_folder) for x in batch_files]
            output_files = [Path(args.output_folder, rel_path) for rel_path in rel_paths]
            for result, output_file in zip(results, output_files):
                save_result(
                    result=result, output_file=output_file, store_probs=args.store_probs
                )
