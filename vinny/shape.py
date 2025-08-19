import time
from typing import Dict, List

import torch
import trimesh
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# The shape generative model, built on a scalable flow-based diffusion transformer
def generate_shape(
        images: Dict[str, Image.Image],
        output_dir: str,
        output_name: str
) -> tuple[List[List[trimesh.Trimesh]], str]:
    print("Loading DiT pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv',
        variant='fp16'
    )

    print("Running shape pipeline...")
    start_time = time.time()
    white_mesh = pipeline(
        image=images,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    output_file = f'{output_dir}/{output_name}.glb'
    white_mesh.export(output_file)
    return white_mesh, output_file
