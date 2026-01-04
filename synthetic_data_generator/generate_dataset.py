import argparse
import os
from tqdm import tqdm

from .generator import SyntheticDataGenerator

def main(args):
    """
    Main script to generate a synthetic dataset.
    It iterates through an input directory of images, applies a number of
    random augmentations to each, and saves the results in an output directory.
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    generator = SyntheticDataGenerator()

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(image_files)} images in {args.input_dir}.")
    print(f"Generating {args.variants} variants for each image.")

    for image_file in tqdm(image_files, desc="Generating Synthetic Data"):
        input_path = os.path.join(args.input_dir, image_file)
        base_name, ext = os.path.splitext(image_file)

        for i in range(args.variants):
            try:
                synthetic_image = generator.generate(input_path)
                output_filename = f"{base_name}_variant_{i+1}{ext}"
                output_path = os.path.join(args.output_dir, output_filename)
                generator.save_image(synthetic_image, output_path)
            except Exception as e:
                print(f"Could not process {image_file}: {e}")

    print(f"\nSynthetic dataset generation complete. Output saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic CCTV-like dataset.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the original, clean images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the generated synthetic images.")
    parser.add_argument("--variants", type=int, default=10, help="Number of synthetic variants to create per input image.")
    
    args = parser.parse_args()
    main(args)
