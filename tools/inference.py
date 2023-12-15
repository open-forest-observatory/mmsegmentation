from mmseg.apis import init_model, inference_model
from pathlib import Path
from imageio import imwrite
import numpy as np
import argparse


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
    if store_probs:
        output_file = output_file.with_suffix(".npy")
        seg_logits = result.seg_logits.data.cpu().numpy()
        seg_logits = np.transpose(seg_logits, (1, 2, 0))
        np.save(output_file, seg_logits)
    else:
        seg = result.pred_sem_seg.data.cpu().numpy()[0].astype(np.uint8)
        output_file = output_file.with_suffix(".png")
        imwrite(output_file, seg)


if __name__ == "__main__":
    args = parse_args()

    files = list(args.image_folder.rglob("*" + args.extension))
    files = list(filter(lambda x: x.is_file(), files))
    args.output_folder.mkdir(exist_ok=True, parents=True)

    model = init_model(str(args.config_path), str(args.checkpoint_path))
    n_files = len(files)
    for i in range(0, len(files), args.batch_size):
        print(f"index: {i}/{n_files}", end="\r")
        batch_files = files[i : i + args.batch_size]

        results = inference_model(model, [str(x) for x in batch_files])

        rel_paths = [x.relative_to(args.image_folder) for x in batch_files]
        output_files = [Path(args.output_folder, rel_path) for rel_path in rel_paths]
        [
            Path(output_file.parent).mkdir(exist_ok=True, parents=True)
            for output_file in output_files
        ]
        [
            save_result(
                result=result, output_file=output_file, store_probs=args.store_probs
            )
            for result, output_file in zip(results, output_files)
        ]
