import argparse
import json
import os

from .bg import BackgroundRemover
from .mesh import TexturedMesh
from .shape import generate_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Json file containing side pictures mapping")
    parser.add_argument("--variant", type=str, required=True, choices=["sdxl", "sd21"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--grid_name", type=str, required=True, help="Filename of the output grid image")
    parser.add_argument("--shape_name", type=str, required=True, help="Filename of the output shape mesh")
    parser.add_argument("--mesh_name", type=str, required=True, help="Filename of the output textured mesh")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Json file {args.input} not found")

    orig_images_path = json.load(open(args.input))
    print("Input images:")
    print(json.dumps(orig_images_path, indent=4))

    print(f"Ensuring output dir {args.output_dir} exists")
    os.makedirs(args.output_dir, exist_ok=True)

    # Remove the background from all pictures
    br = BackgroundRemover()
    nobg_images = br.remove_bg(
        files=orig_images_path,
        output_dir=args.output_dir
    )

    # Generate the white mesh based on the background-free images
    shape, shape_file = generate_shape(
        images=nobg_images,
        output_dir=args.output_dir,
        output_name=args.shape_name
    )
    # shape_file = f"{os.path.join(args.output_dir, args.shape_name)}.glb"

    # Infer all view angles based on the front view
    # override some of the grid images with the known images
    # use the white mesh and the grid to generate the textured mesh
    text = "high quality"
    reference_conditioning_scale = 1.0
    seed = 42
    output_images_grid = f"{os.path.join(args.output_dir, args.grid_name)}.png"

    mesh = TexturedMesh(args.variant)
    images_grid = mesh.generate_mv_grid(
        mesh=shape_file,
        text=text,
        image=nobg_images["front"],
        seed=seed,
        reference_conditioning_scale=reference_conditioning_scale
    )
    mesh.override_grid(
        images=images_grid,
        orig=nobg_images,
        grid_path=output_images_grid
    )
    mesh.generate_texture(
        mesh_path=shape_file,
        save_dir=args.output_dir,
        save_name=args.mesh_name,
        rgb_path=output_images_grid
    )
