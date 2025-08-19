import argparse
import os

from .bg import BackgroundRemover
from .shape import generate_shape

orig_images_path = {
    "front": "Jeans_1_0.jpg",
    "left": "Jeans_1_1.jpg",
    "back": "Jeans_1_2.jpg",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--shape_name", type=str, required=True)
    args = parser.parse_args()

    print(f"Ensuring output dir {args.output_dir} exists")
    os.makedirs(args.output_dir, exist_ok=True)

    br = BackgroundRemover()
    nobg_images = {}
    for side, path in orig_images_path.items():
        fullpath = os.path.join(args.input_dir, path)
        nobg_image = br.remove_bg(fullpath, args.output_dir)
        nobg_images[side] = nobg_image

    generate_shape(nobg_images, args.output_dir, args.shape_name)
