import functools
import gc
import json
from typing import Any

import torch

from .cuda import ensure_cuda
from .bg import BackgroundRemover
from .video import get_frames
from .shape import Shape
from .grid import MVGrid
from .mesh import TexturedMesh


class Vinny:
    def __init__(self, input: Any, variant: str, output_dir: str):
        self.input = input
        self.variant = variant
        self.output_dir = output_dir

        self.device = ensure_cuda()

    @staticmethod
    def checkpoint(f):
        @functools.wraps(f)
        def _wrapper(self, *args, **kwargs):
            f(self, *args, **kwargs)
            self._clear_mem()
            self._save_json()

        return _wrapper

    def __call__(self, *args, **kwargs):
        self._extract_frames_from_videos()
        self._remove_bg_from_all()
        self._generate_shape_for_all()
        self._generate_mv_grid_for_all()
        self._generate_texture_for_all()

    @checkpoint
    def _extract_frames_from_videos(self):
        for prefix, item in self.input.items():
            if item["type"] == "video":
                images_path = get_frames(
                    video_path=item["orig_video"],
                    prefix=prefix,
                    output_dir=self.output_dir
                )
                item["orig_images"] = images_path

    # Remove the background from all pictures
    @checkpoint
    def _remove_bg_from_all(self):
        br = BackgroundRemover()
        for prefix, item in self.input.items():
            nobg_images = br.remove_bg(
                files=item["orig_images"],
                output_dir=self.output_dir
            )
            item["nobg_images"] = nobg_images

    # Generate the white mesh based on the background-free images
    @checkpoint
    def _generate_shape_for_all(self):
        shape = Shape()
        for prefix, item in self.input.items():
            shape_file = f'{self.output_dir}/{prefix}_shape.glb'
            shape.generate(
                images_path=item["nobg_images"],
                output_file=shape_file
            )
            item["shape"] = shape_file

    # Infer all view angles based on the front view
    # override some of the grid images with the known images
    # use the white mesh and the grid to generate the textured mesh
    @checkpoint
    def _generate_mv_grid_for_all(self):
        grid = MVGrid(device=self.device, variant=self.variant)

        text = "high quality"
        reference_conditioning_scale = 1.0
        seed = 42

        for prefix, item in self.input.items():
            output_images_grid = f"{self.output_dir}/{prefix}_grid.png"
            images_grid = grid.generate_mv_grid(
                mesh=item["shape"],
                text=text,
                image=item["nobg_images"]["front"],
                seed=seed,
                reference_conditioning_scale=reference_conditioning_scale
            )
            grid.override_grid(
                images=images_grid,
                overrides=item["nobg_images"],
                grid_path=output_images_grid
            )
            item["mv_grid"] = output_images_grid

    @checkpoint
    def _generate_texture_for_all(self):
        mesh = TexturedMesh(device=self.device, variant=self.variant)
        for prefix, item in self.input.items():
            mesh_path = mesh.generate_texture(
                mesh_path=item["shape"],
                save_dir=self.output_dir,
                save_name=f"{prefix}_mesh",
                rgb_path=item["mv_grid"]
            )
            item["mesh"] = mesh_path.shaded_model_save_path

    def _save_json(self):
        print("Saving JSON...")
        with open(f'{self.output_dir}/output.json', 'w') as f:
            json.dump(self.input, f, indent=2)

    @staticmethod
    def _clear_mem():
        print("Clearing memory...")
        torch.cuda.empty_cache()
        gc.collect()
