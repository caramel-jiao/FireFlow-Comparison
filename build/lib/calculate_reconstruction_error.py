import os
import re
import numpy as np
from PIL import Image
import json
import math

def calculate_mse(image1, image2):
    """
    Calculate the Mean Squared Error (MSE) between two images.
    """
    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)
    mse = np.mean((arr1 - arr2) ** 2)
    return math.sqrt(float(mse))

def extract_steps_from_filename(filename):
    """
    Extract steps (X) from filename pattern 'steps_X'.
    """
    match = re.search(r'steps_(\d+)', filename)
    return int(match.group(1)) if match else None

def process_images(folder_path, reference_image_path, output_path):
    """
    Calculate MSE between reference image and all images in the folder.
    Save results as a dictionary: {steps: mse}.
    """
    # Load reference image
    reference_image = Image.open(reference_image_path).convert('RGB')
    results = {}

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(folder_path, filename)
            steps = extract_steps_from_filename(filename)

            if steps is not None:
                # Load and calculate MSE
                image = Image.open(image_path).convert('RGB')
                mse = calculate_mse(reference_image, image)
                results[steps] = mse

    # Save results to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"MSE results saved to {output_path}")

# Example usage
reference_image_path = "/data/hexiangyu/fireflow/src/examples/source/boy.jpg"

folder_path = "/data/hexiangyu/fireflow/src/examples/rf_solver_inversion_result/dog"
output_path = "./rf_solver_results.json"
process_images(folder_path, reference_image_path, output_path)

folder_path = "/data/hexiangyu/fireflow/src/examples/reflow_inversion_result/dog"
output_path = "./reflow_results.json"
process_images(folder_path, reference_image_path, output_path)

folder_path = "/data/hexiangyu/fireflow/src/examples/fireflow_inversion_result/dog"
output_path = "./rf_modified_midpoint_results.json"
process_images(folder_path, reference_image_path, output_path)