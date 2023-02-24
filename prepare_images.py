import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def process_images(args):
    images_folder = Path(args.folder_path)
    output_folder = Path(f'data/{args.name}/images')
    output_folder.mkdir(parents=True, exist_ok=False)

    input_files = sorted(
        list(images_folder.glob("*.jpg")) + list(images_folder.glob("*.jpeg")) + list(images_folder.glob("*.png")))
    for i, image_path in tqdm(enumerate(input_files)):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if args.method == "crop":  # apply center crop to all images.
            min_size = min(w, h)
            left = int((w - min_size) / 2)
            top = int((h - min_size) / 2)
            right = int((w + min_size) / 2)
            bottom = int((h + min_size) / 2)
            image = image.crop((left, top, right, bottom))
        elif args.method == "pad":  # apply border padding to all images
            max_size = max(w, h)
            if w < h:
                pad = (max_size - w)
                pad_left = int(np.ceil(pad / 2))
                pad_right = pad - pad_left
                padding = (pad_left, 0, pad_right, 0)
            else:
                pad = (max_size - h)
                pad_top = int(np.ceil(pad / 2))
                pad_bottom = pad - pad_top
                padding = (0, pad_top, 0, pad_bottom)
            if pad > 0:
                image = transforms.Pad(padding=padding, padding_mode="edge")(image)
        else:
            raise ValueError("'method' argument should be one of [crop, pad].")

        # resize image
        w, h = image.size
        if w != args.resolution:
            image = image.resize((args.resolution, args.resolution), Image.ANTIALIAS)

        # save images
        image.save(f"{output_folder}/image_{i:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Square and resize images.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing the image set.')
    parser.add_argument('--name', type=str, required=True, help='Name of image set.')
    parser.add_argument('--method', default='crop', type=str, help='crop | pad. crop will apply center crop to '
                                                                   'all images ; pad will pad all images to square.')
    parser.add_argument('--resolution', default=256, type=int, help='Final resolution of images.')
    args = parser.parse_args()

    process_images(args)
