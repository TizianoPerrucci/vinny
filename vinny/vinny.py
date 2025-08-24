import gc
import time
import torch

from .bg import BackgroundRemover
from .shape import Shape
from .mesh import TexturedMesh


class Vinny:
    def __init__(self, variant: str):
        self.br = BackgroundRemover()
        self.shape = Shape()
        self.mesh = TexturedMesh(variant)

    def process(self, orig_images_path: dict[str, str], prefix: str, output_dir: str):
        # Remove the background from all pictures
        nobg_images = self.br.remove_bg(
            files=orig_images_path,
            output_dir=output_dir
        )

        self.clear_mem()

        # Generate the white mesh based on the background-free images
        shape, shape_file = self.shape.generate(
            images=nobg_images,
            output_file=f'{output_dir}/{prefix}_shape.glb'
        )
        # shape_file = f'{output_dir}/{prefix}_shape.glb'

        self.clear_mem()

        # Infer all view angles based on the front view
        # override some of the grid images with the known images
        # use the white mesh and the grid to generate the textured mesh
        text = "high quality"
        reference_conditioning_scale = 1.0
        seed = 42
        output_images_grid = f"{output_dir}/{prefix}_grid.png"

        images_grid = self.mesh.generate_mv_grid(
            mesh=shape_file,
            text=text,
            image=nobg_images["front"],
            seed=seed,
            reference_conditioning_scale=reference_conditioning_scale
        )
        self.mesh.override_grid(
            images=images_grid,
            orig=nobg_images,
            grid_path=output_images_grid
        )

        self.clear_mem()

        self.mesh.generate_texture(
            mesh_path=shape_file,
            save_dir=output_dir,
            save_name="mesh",
            rgb_path=output_images_grid
        )

    @staticmethod
    def clear_mem():
        torch.cuda.empty_cache()
        gc.collect()

        time.sleep(10)
