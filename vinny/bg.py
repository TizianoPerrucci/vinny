import os
from typing import Dict

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover:
    def __init__(self):
        self.session = new_session(model_name="u2net")

    def remove_bg(self, files: Dict[str, str], output_dir: str) -> Dict[str, str]:
        nobg_images: Dict[str, str] = {}
        for side, image_path in files.items():
            print(f"Processing image {image_path}...")
            output_name = os.path.splitext(os.path.basename(image_path))[0] + "_nobg.png"
            output_file = os.path.join(output_dir, output_name)

            # Remove background and save
            nobg_img = remove(
                data=Image.open(image_path),
                session=self.session,
                bgcolor=(255, 255, 255, 0),
                post_process_mask=True
            )
            nobg_img.save(output_file)
            nobg_images[side] = output_file
            print(f"Saved modified image to {output_file}")

        return nobg_images
