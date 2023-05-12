# temporary file for demos
# see diffusion/scripts/generate_from_file.py

import argparse


def main(image_path, mask_path):
    print("Image path:", image_path)
    print("Mask path:", mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="path to the image file")
    parser.add_argument("mask_path", help="path to the mask file")
    args = parser.parse_args()

    main(args.image_path, args.mask_path)
