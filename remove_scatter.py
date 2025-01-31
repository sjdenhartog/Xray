import numpy as np
from PIL import Image
from pathlib import Path
import re
import os

all_numbers = np.arange(50,220) # all frames that are taken into account
path_to_date = Path(r'D:\XRay\2024-11-28 Rik en Sam')

def average_intensity(path, all_numbers):
    image_count = len(all_numbers)
    average_array = None
    # Loop through each image
    for num in all_numbers:
        image_path = path / f'img_{num}.tif'
        with Image.open(image_path) as img:
            img_array = np.array(img, dtype=np.float64)  # Convert image to NumPy array
            if average_array is None:
                average_array = np.zeros_like(img_array)  # Initialize accumulator
            average_array += img_array  # Add pixel values

    # Calculate average
    average_array /= image_count

    image_array = average_array
    return image_array

def remove_scatter(path_to_cam, all_numbers, scatter_array, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(path_to_cam)
    # Loop through each image
    for num in all_numbers:
        image_path = path_to_cam / f'img_{num}.tif'
        
        print(f'img_{num}.tif')
        with Image.open(image_path) as img:
            img_array = np.array(img, dtype=np.float64)  # Convert image to NumPy array
            # Subtract scatter array
            corrected_img_array = img_array - scatter_array
            
            # convert unit16 for more values in the grayscale (so not up to 255)
            corrected_img_array = corrected_img_array.astype(np.uint16)
            
            # convert the corrected array back to an image
            corrected_img = Image.fromarray(corrected_img_array)

            corrected_img.save(output_dir / f'img_{num}.tif')

        
def scattercorrected_output_dir(path, camera, gasflow):
    if gasflow == 0:
        # for 0lmin
        path_to_prepro_folder_0lmin = re.sub(r'_scatter.*', '', str(path)) 

        # Convert back to Path object
        path_to_prepro_folder_0lmin = Path(path_to_prepro_folder_0lmin)

        path_to_prepro_cam = path_to_prepro_folder_0lmin / camera
        
        output_dir = re.sub(r'preprocessed.', 'scattercorrected_', str(path_to_prepro_cam))

    elif gasflow == 80: # for 80lmin
        path_to_prepro_folder_80lmin = re.sub(r'0lmin_scatter.*', '80lmin', str(path)) 

        # Convert back to Path object
        path_to_prepro_folder_80lmin = Path(path_to_prepro_folder_80lmin)

        path_to_prepro_cam = path_to_prepro_folder_80lmin / camera
        
        output_dir = re.sub(r'preprocessed.', 'scattercorrected_', str(path_to_prepro_cam))
    output_dir = Path(output_dir)
    return path_to_prepro_cam, output_dir

for folder in path_to_date.iterdir():
    if folder.is_dir(): 
        if folder.name.startswith('preprocessed_') and folder.name.endswith('_D1'): 
            print(folder.name)
            camera = 'camera 1'
            path = path_to_date / folder 
            path_to_camera = path / camera
            scatter_array_cam1 = average_intensity(path_to_camera, all_numbers)
            
            # 0 lmin
            path_to_prepro_cam_0, output_0 = scattercorrected_output_dir(path, camera, 0)
            last_file = len(list(path_to_prepro_cam_0.glob("*.tif")))
            remove_scatter(path_to_prepro_cam_0, all_numbers=range(50, last_file), scatter_array=scatter_array_cam1, output_dir=output_0)

            # 80 lmin
            path_to_prepro_cam_80, output_80 = scattercorrected_output_dir(path, camera, 80)
            last_file = len(list(path_to_prepro_cam_80.glob("*.tif")))
            remove_scatter(path_to_prepro_cam_80, all_numbers=range(50, last_file), scatter_array=scatter_array_cam1, output_dir=output_80)

        elif folder.name.startswith('preprocessed_') and folder.name.endswith('_D2'):  
            camera = 'camera 2'
            path = path_to_date / folder 
            path_to_camera = path / camera
            scatter_array_cam2 = average_intensity(path_to_camera, all_numbers)

            # 0 lmin
            path_to_prepro_cam_0, output_0 = scattercorrected_output_dir(path, camera, 0)
            last_file = len(list(path_to_prepro_cam_0.glob("*.tif")))
            remove_scatter(path_to_prepro_cam_0, all_numbers=range(50, last_file), scatter_array=scatter_array_cam2, output_dir=output_0)

            # 80 lmin
            path_to_prepro_cam_80, output_80 = scattercorrected_output_dir(path, camera, 80)
            last_file = len(list(path_to_prepro_cam_80.glob("*.tif")))
            remove_scatter(path_to_prepro_cam_80, all_numbers=range(50, last_file), scatter_array=scatter_array_cam2, output_dir=output_80)

        elif folder.name.startswith('preprocessed_') and folder.name.endswith('_D3'): 
            camera = 'camera 3'
            path = path_to_date / folder 
            path_to_camera = path / camera
            scatter_array_cam3 = average_intensity(path_to_camera, all_numbers)

           # 0 lmin
            path_to_prepro_cam_0, output_0 = scattercorrected_output_dir(path, camera, 0)
            last_file = len(list(path_to_prepro_cam_0.glob("*.tif")))
            remove_scatter(path_to_prepro_cam_0, all_numbers=range(50, last_file), scatter_array=scatter_array_cam3, output_dir=output_0)

            # 80 lmin
            path_to_prepro_cam_80, output_80 = scattercorrected_output_dir(path, camera, 80)
            last_file = len(list(path_to_prepro_cam_80.glob("*.tif")))
            remove_scatter(path_to_prepro_cam_80, all_numbers=range(50, last_file), scatter_array=scatter_array_cam3, output_dir=output_80)