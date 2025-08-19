import argparse
import os
from typing import Dict

from PIL import Image

from .bg import BackgroundRemover
from .mesh import TexturedMesh
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
    parser.add_argument("--variant", type=str, required=True, choices=["sdxl", "sd21"])
    parser.add_argument("--mesh_name", type=str, required=True)
    args = parser.parse_args()

    print(f"Ensuring output dir {args.output_dir} exists")
    os.makedirs(args.output_dir, exist_ok=True)

    # Remove the background from all pictures
    br = BackgroundRemover()
    nobg_images: Dict[str, Image.Image] = {}
    for side, path in orig_images_path.items():
        fullpath = os.path.join(args.input_dir, path)
        nobg_image = br.remove_bg(image_path=fullpath, output_dir=args.output_dir)
        nobg_images[side] = nobg_image

    # Generate the white mesh based on the background-free images
    shape, shape_file = generate_shape(nobg_images, args.output_dir, args.shape_name)

    # Infer all view angles based on the front view
    # override some of the grid images with the known images
    # use the white mesh and the grid to generate the textured mesh
    text = "high quality"
    reference_conditioning_scale = 1.0
    seed = 42
    output_images_grid = os.path.join(args.output_dir, "grid.png")

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
        orig=list(nobg_images.values()),
        grid_path=output_images_grid
    )
    mesh.generate_texture(
        mesh_path=shape_file,
        save_dir=args.output_dir,
        save_name=args.mesh_name,
        rgb_path=output_images_grid
    )
