from mvadapter.pipelines.pipeline_texture import TexturePipeline, ModProcessConfig


class TexturedMesh:
    def __init__(self, device: str, variant: str):
        if variant == "sdxl":
            self.uv_size = 4096
        elif variant == "sd21":
            self.uv_size = 2048
        else:
            raise ValueError(f"MVAdapter variant {variant} invalid.")

        print("Loading texture pipeline...")
        self.texture_pipe = TexturePipeline(
            upscaler_ckpt_path="/app/checkpoints/RealESRGAN_x2plus.pth",
            inpaint_ckpt_path="/app/checkpoints/big-lama.pt",
            device=device,
        )

    def generate_texture(self, mesh_path: str, save_dir: str, save_name: str, rgb_path: str) -> str:
        print("Running texture pipeline...")
        textured_glb_path = self.texture_pipe(
            mesh_path=mesh_path,
            save_dir=save_dir,
            save_name=save_name,
            uv_unwarp=True,
            preprocess_mesh=True,
            uv_size=self.uv_size,
            rgb_path=rgb_path,
            rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
            camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            debug_mode=True
        )

        print(f"Textured mesh saved to {textured_glb_path}")
        return textured_glb_path
