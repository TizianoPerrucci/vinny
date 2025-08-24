import cv2
import numpy as np
from PIL import Image


def get_frames(video_path: str, prefix: str, output_dir: str) -> dict[str, str]:
    print(f"Extracting video frames from {video_path}...")
    videocap = cv2.VideoCapture(video_path)
    fps = videocap.get(cv2.CAP_PROP_FPS)
    frame_count = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds = frame_count / fps

    print(f"frames: {frame_count}, fps: {fps}, duration: {seconds}")

    # take evenly spaced frames, last one won't count
    frame_indices = np.round(np.linspace(0, frame_count, 5))
    print(f"Frame indices: {frame_indices}")

    # iterate frame indices and grab the frame
    sides = ["front", "left", "back", "right"]
    sideidx = 0
    images: dict[str, str] = {}
    for frame_number in frame_indices:
        videocap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = videocap.read()  # read video

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).convert('RGB')

        output_file = f"{output_dir}/{prefix}_{str(frame_number)}.png"
        print(f"Saving frame {frame_number} to {output_file}")
        img.save(output_file)

        images[sides[sideidx]] = output_file
        sideidx = sideidx + 1

    videocap.release()
    return images
