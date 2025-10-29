
import argparse
import json
import os
from glob import glob

import numpy as np
from src.change_detection import compute_difference
from src.image_processing import align_images, load_image, resize_image
from src.visualization import create_heatmap, save_image


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visual difference engine.")
    parser.add_argument("--input-dir", help="Path to the directory of images.")
    parser.add_argument("--image1", help="Path to the first image.")
    parser.add_argument("--image2", help="Path to the second image.")
    parser.add_argument("--output-dir", help="Path to the output directory.")
    parser.add_argument("--output", help="Path to the output image.")
    parser.add_argument("--report", help="Path to the output JSON report.")
    args = parser.parse_args()

    if args.input_dir:
        if not args.output_dir:
            print("Please specify an output directory using --output-dir.")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        images = sorted(glob(os.path.join(args.input_dir, "*.png"))) + sorted(
            glob(os.path.join(args.input_dir, "*.jpg"))
        )

        for i in range(len(images) - 1):
            # Load images
            image1 = load_image(images[i])
            image2 = load_image(images[i + 1])

            if image1.shape != image2.shape:
                image2 = resize_image(image2, (image1.shape[1], image1.shape[0]))

            # Align images
            aligned_image1, aligned_image2 = align_images(image1, image2)

            # Compute difference
            diff = compute_difference(aligned_image1, aligned_image2)
            print(f"Diff min: {np.min(diff)}, max: {np.max(diff)}, mean: {np.mean(diff)}")

            # Create heatmap
            heatmap = create_heatmap(diff)
            print(f"Heatmap min: {np.min(heatmap)}, max: {np.max(heatmap)}, mean: {np.mean(heatmap)}")

            # Save heatmap
            output_path = os.path.join(
                args.output_dir, f"diff_{i}_{i+1}.png"
            )
            save_image(heatmap, output_path)

            # Save report
            report_path = os.path.join(
                args.output_dir, f"report_{i}_{i+1}.json"
            )
            threshold = 30  # Adjust threshold for change detection
            change_percentage = np.count_nonzero(diff > threshold) / diff.size * 100
            report = {
                "change_percentage": change_percentage
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)

    elif args.image1 and args.image2:
        if not args.output:
            print("Please specify an output path using --output.")
            return

        # Load images
        image1 = load_image(args.image1)
        image2 = load_image(args.image2)

        if image1.shape != image2.shape:
            image2 = resize_image(image2, (image1.shape[1], image1.shape[0]))

        # Align images
        aligned_image1, aligned_image2 = align_images(image1, image2)

        # Compute difference
        diff = compute_difference(aligned_image1, aligned_image2)
        print(f"Diff min: {np.min(diff)}, max: {np.max(diff)}, mean: {np.mean(diff)}")

        # Create heatmap
        heatmap = create_heatmap(diff)
        print(f"Heatmap min: {np.min(heatmap)}, max: {np.max(heatmap)}, mean: {np.mean(heatmap)}")

        # Save heatmap
        save_image(heatmap, args.output)

        # Save report
        if args.report:
            threshold = 30  # Adjust threshold for change detection
            change_percentage = np.count_nonzero(diff > threshold) / diff.size * 100
            report = {
                "change_percentage": change_percentage
            }
            with open(args.report, "w") as f:
                json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()
