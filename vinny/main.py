import argparse
import json
import os

from .vinny import Vinny

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Json file containing all items to process")
    parser.add_argument("--variant", type=str, required=True, choices=["sdxl", "sd21"])
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Ensuring output dir {args.output_dir} exists")
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Json file {args.input} not found")

    input = json.load(open(args.input))

    vinny = Vinny(
        input=input,
        variant=args.variant,
        output_dir=args.output_dir
    )
    vinny()
