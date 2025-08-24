from typing import Optional, Type, List, Union, Dict

import torch
import numpy as np
from PIL import Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    LCMScheduler
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.pipelines.pipeline_texture import TexturePipeline, ModProcessConfig
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    get_orthogonal_camera,
    load_mesh,
    render,
)

from vinny.cuda import ensure_cuda


class TexturedMesh:
    def __init__(self, variant: str):
        self.device = ensure_cuda()

        if variant == "sdxl":
            self.pipeline_class = MVAdapterI2MVSDXLPipeline
            self.base_model = "stabilityai/stable-diffusion-xl-base-1.0"
            self.vae_model = "madebyollin/sdxl-vae-fp16-fix"
            self.adapter_weight_name = "mvadapter_ig2mv_sdxl.safetensors"
            self.height = self.width = 768
            self.uv_size = 4096
        elif variant == "sd21":
            self.pipeline_class = MVAdapterI2MVSDPipeline
            self.base_model = "stabilityai/stable-diffusion-2-1-base"
            self.vae_model = None
            self.adapter_weight_name = "mvadapter_ig2mv_sd21.safetensors"
            self.height = self.width = 512
            self.uv_size = 2048
        else:
            raise ValueError(f"MVAdapter variant {variant} invalid.")

        self.num_views = 6
        self.adapter_path = "huanngzh/mv-adapter"

        print("Loading MV-Adapter pipeline...")
        self.mv_pipe = self._prepare_pipeline(
            pipeline_class=self.pipeline_class,
            base_model=self.base_model,
            vae_model=self.vae_model,
            unet_model=None,
            lora_model=None,
            adapter_path=self.adapter_path,
            adapter_weight_name=self.adapter_weight_name,
            scheduler=None,
            dtype=torch.float16,
        )

        print("Loading texture pipeline...")
        self.texture_pipe = TexturePipeline(
            upscaler_ckpt_path="/app/checkpoints/RealESRGAN_x2plus.pth",
            inpaint_ckpt_path="/app/checkpoints/big-lama.pt",
            device=self.device,
        )

    def _prepare_pipeline(
            self,
            pipeline_class: Type[StableDiffusionPipeline],
            base_model: str,
            vae_model: str,
            unet_model: Optional[str],
            lora_model: Optional[str],
            adapter_path: str,
            adapter_weight_name: str,
            scheduler: Optional[str],
            dtype: torch._C.dtype,
    ) -> StableDiffusionPipeline:
        # Load vae and unet if provided
        pipe_kwargs = {}
        if vae_model is not None:
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
        if unet_model is not None:
            pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

        # Prepare pipeline
        pipe = pipeline_class.from_pretrained(base_model, **pipe_kwargs)

        # Load scheduler if provided
        scheduler_class = None
        if scheduler == "ddpm":
            scheduler_class = DDPMScheduler
        elif scheduler == "lcm":
            scheduler_class = LCMScheduler

        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class,
        )
        pipe.init_custom_adapter(
            num_views=self.num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
        )
        pipe.load_custom_adapter(
            adapter_path, weight_name=adapter_weight_name
        )

        pipe.to(device=self.device, dtype=dtype)
        pipe.cond_encoder.to(device=self.device, dtype=dtype)

        # load lora if provided
        if lora_model is not None:
            model_, name_ = lora_model.rsplit("/", 1)
            pipe.load_lora_weights(model_, weight_name=name_)

        # vae slicing for lower memory usage
        pipe.enable_vae_slicing()

        return pipe

    @staticmethod
    def _preprocess_image(image: Image.Image, height: int, width: int):
        image = np.array(image)
        alpha = image[..., 3] > 0
        H, W = alpha.shape
        # get the bounding box of alpha
        y, x = np.where(alpha)
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
        image_center = image[y0:y1, x0:x1]
        # resize the longer side to H * 0.9
        H, W, _ = image_center.shape
        if H > W:
            W = int(W * (height * 0.9) / H)
            H = int(height * 0.9)
        else:
            H = int(H * (width * 0.9) / W)
            W = int(width * 0.9)
        image_center = np.array(Image.fromarray(image_center).resize((W, H)))
        # pad to H, W
        start_h = (height - H) // 2
        start_w = (width - W) // 2
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[start_h: start_h + H, start_w: start_w + W] = image_center
        image = image.astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        return image

    def _run_pipeline(
            self,
            pipe: StableDiffusionPipeline,
            mesh_path: str,
            text: str,
            image: Union[Image.Image, str],
            num_inference_steps: int,
            guidance_scale: float,
            seed: int,
            reference_conditioning_scale=1.0,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            lora_scale=1.0,
    ):
        # Prepare cameras
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
            distance=[1.8] * self.num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            device=self.device,
        )
        ctx = NVDiffRastContextWrapper(device=self.device)

        mesh = load_mesh(mesh_path, rescale=True, device=self.device)
        render_out = render(
            ctx,
            mesh,
            cameras,
            height=self.height,
            width=self.width,
            render_attr=False,
            normal_background=0.0,
        )
        pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
        normal_images = tensor_to_image(
            (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
        )
        control_images = (
            torch.cat(
                [
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

        # Prepare image
        reference_image = Image.open(image) if isinstance(image, str) else image
        if reference_image.mode == "RGBA":
            reference_image = self._preprocess_image(reference_image, self.height, self.width)

        pipe_kwargs = {}
        if seed != -1 and isinstance(seed, int):
            pipe_kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        images = pipe(
            text,
            height=self.height,
            width=self.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=self.num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            reference_image=reference_image,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"scale": lora_scale},
            **pipe_kwargs,
        ).images

        return images, pos_images, normal_images, reference_image

    def generate_mv_grid(
            self,
            mesh: str,
            text: str,
            image: Union[Image.Image, str],
            seed: int,
            reference_conditioning_scale: float,
    ):
        print("Running MV-Adapter pipeline...")
        images, _, _, _ = self._run_pipeline(
            pipe=self.mv_pipe,
            mesh_path=mesh,
            text=text,
            image=image,
            num_inference_steps=50,
            guidance_scale=3.0,
            seed=seed,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        )
        print("Grid images generated")
        return images

    def override_grid(self, images: List[Image.Image], orig: Dict[str, Image.Image], grid_path: str):
        (w, h) = images[0].size
        print(f"Overriding first {len(orig)} images, based on size: w={w} h={h}")

        side_to_idx = {"front": 0, "left": 1, "back": 2, "right": 3}
        for side, img in orig.items():
            print(f"PreProcessing {img}...")
            images[side_to_idx[side]] = self._preprocess_image(img, h, w)

        print(f"Saving images grid to {grid_path} ...")
        make_image_grid(images, rows=1).save(grid_path)
        return orig

    def generate_texture(self, mesh_path: str, save_dir: str, save_name: str, rgb_path: str):
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
