import time
from typing import Dict, List

import torch
import trimesh
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


class Shape:
    def __init__(self):
        print("Loading DiT pipeline...")
        # The multi-view shape generative model, built on a scalable flow-based diffusion transformer
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path='tencent/Hunyuan3D-2mv',
            subfolder='hunyuan3d-dit-v2-mv',
        )

    def generate(
            self,
            images_path: Dict[str, str],
            output_file: str
    ) -> tuple[List[List[trimesh.Trimesh]], str]:
        print("Running shape pipeline...")
        start_time = time.time()
        images = {k: Image.open(v) for k, v in images_path.items()}
        white_mesh = self.pipeline(
            image=images,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]
        print("--- %s seconds ---" % (time.time() - start_time))
        white_mesh.export(output_file)
        print(f"Shape saved to {output_file}")
        return white_mesh, output_file
