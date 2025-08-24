import argparse
import json
import os

from .vinny import Vinny
from .video import get_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Json file containing all items to process")
    parser.add_argument("--variant", type=str, required=True, choices=["sdxl", "sd21"])
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Json file {args.input} not found")

    print(f"Ensuring output dir {args.output_dir} exists")
    os.makedirs(args.output_dir, exist_ok=True)

    vinny = Vinny(variant=args.variant)

    input = json.load(open(args.input))
    for prefix, item in input.items():
        print(f"Processing item {prefix}...")
        print(json.dumps(item, indent=4))

        item_type = item.pop("type")
        if item_type == "images":
            images_path = item
        elif item_type == "video":
            images_path = get_frames(
                video_path=item["src"],
                prefix=prefix,
                output_dir=args.output_dir
            )

        vinny.process(
            orig_images_path=images_path,
            prefix=prefix,
            output_dir=args.output_dir,
        )
