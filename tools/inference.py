from mmseg.apis import init_model, inference_model
from pathlib import Path
from imageio import imwrite
from PIL import Image
import numpy as np
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--store-probs", action="store_true")
    parser.add_argument("--extension", default="")

    args = parser.parse_args()
    return args


def save_result(result, output_file, store_probs, orientation):
    # Extract either the logits or highest confidence class
    if store_probs:
        seg = result.seg_logits.data.cpu().numpy()
        seg = np.transpose(seg, (1, 2, 0))
    else:
        seg = result.pred_sem_seg.data.cpu().numpy()[0].astype(np.uint8)

    # Flip the image if it was originally upsidedown
    if orientation == 3:
        print(f"Flipping {output_file}")
        seg = np.flip(seg, (0, 1))

    # Create the the folder for the file
    Path(output_file.parent).mkdir(exist_ok=True, parents=True)

    # Write as a different file type based on whether it's two or three dimensional
    if store_probs:
        output_file = output_file.with_suffix(".npy")
        np.save(output_file, seg)
    else:
        output_file = output_file.with_suffix(".png")
        imwrite(output_file, seg)


def get_image_shape_orientation(file):
    """Return the (h, w) tuple of image shape if it's a image, otherwise None"""
    try:
        image = Image.open(file)
        size = image.size
        # Read the image exif
        exif_dict = image.getexif()
        # If the orientation flag is present, take that value. Otherwise use the default of 1.
        orientation = exif_dict.get(274, 1)
        return (size, orientation)
    except:
        return None


def main(
    config_path,
    checkpoint_path,
    image_folder,
    output_folder,
    batch_size=4,
    store_probs=False,
    extension="",
):
    # Get all files
    search_string = str(Path(image_folder, "**" + extension))
    all_files = glob.glob(search_string, recursive=True)
    all_files = list(filter(lambda x: Path(x).is_file(), all_files))
    # Get the shapes and orientation of all images. Will be None if not an image
    # This step may be slow
    print("Reading image shapes and orientations. This step can be slow")
    image_shapes_and_orientation = [
        get_image_shape_orientation(file) for file in all_files
    ]
    print("Done reading shapes and orientations")
    # Merge the shapes with the paths
    shape_orientation_file_list = zip(image_shapes_and_orientation, all_files)
    # Filter out the tuples that don't correspond to an image
    shape_orientation_file_list = list(
        filter(lambda x: x[0] is not None, shape_orientation_file_list)
    )
    # Get the unique shapes across all images
    unique_shapes = np.unique(
        [shape_file[0][0] for shape_file in shape_orientation_file_list], axis=0
    )
    # Convert back into tuples
    unique_shapes = [tuple(unique_shape) for unique_shape in unique_shapes]
    model = init_model(str(config_path), str(checkpoint_path))

    print(f"Unique image shapes are {unique_shapes}")

    for unique_shape in unique_shapes:
        print(f"Processing images with {unique_shape} shape")
        # Extract the filenames corresponding to images of that shape
        matching = list(
            filter(lambda x: x[0][0] == unique_shape, shape_orientation_file_list)
        )

        n_files = len(matching)
        for i in range(0, len(matching), batch_size):
            print(f"index: {i}/{n_files}", end="\r")
            batching = matching[i : i + batch_size]

            # Get the file names
            files = [x[1] for x in batching]
            orientations = [x[0][1] for x in batching]

            results = inference_model(model, [str(x) for x in files])

            rel_paths = [Path(x).relative_to(Path(image_folder)) for x in files]
            output_files = [Path(output_folder, rel_path) for rel_path in rel_paths]
            for result, output_file, orientation in zip(
                results, output_files, orientations
            ):
                save_result(
                    result=result,
                    output_file=output_file,
                    store_probs=store_probs,
                    orientation=orientation,
                )


if __name__ == "__main__":
    args = parse_args()

    main(**args.__dict__)
