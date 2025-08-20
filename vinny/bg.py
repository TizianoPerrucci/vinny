import os
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from .cuda import ensure_cuda


class BackgroundRemover:
    def __init__(self):
        device = ensure_cuda()

        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.rm_fn = lambda img: self._rm_bg(img, birefnet, transform_image, device)

    def _rm_bg(self, image: Image.Image, net, transform, device) -> Image.Image:
        image_size = image.size
        input_images = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = net(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image

    def remove_bg(self, files: Dict[str, str], output_dir: str) -> Dict[str, Image.Image]:
        nobg_images: Dict[str, Image.Image] = {}
        for side, image_path in files.items():
            print(f"Processing image {image_path}...")
            output_name = os.path.splitext(os.path.basename(image_path))[0] + "_nobg_orig.png"
            output_file = os.path.join(output_dir, output_name)

            # Remove background and save
            nobg_img = self.rm_fn(Image.open(image_path))
            nobg_images[side] = nobg_img
            nobg_img.save(output_file)
            print(f"Saved modified image to {output_file}")

        return nobg_images
