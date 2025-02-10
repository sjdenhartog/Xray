import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

folder_path1 = r"D:\XRay\2024-11-27 Rik en Sam\BHC_IPA_00gl_0lmin\camera 1"  
folder_path2 = r'D:\XRay\2024-11-27 Rik en Sam\scattercorrected_IPA_00gl_0lmin\camera 1'

all_numbers = np.arange(50,220)
file1_intensity = average_intensity(Path(folder_path1), all_numbers)
print(f'average pixel value folder1 = {file1_intensity.mean()}')
file2_intensity = average_intensity(Path(folder_path2), all_numbers)
print(f'average pixel value folder2 = {file2_intensity.mean()}')
plt.imshow(file2_intensity-(file2_intensity-file1_intensity))
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # For reading .tif images

# Folder path containing the .tif files
folder_path = r'D:\XRay\2024-11-27 Rik en Sam\BHC_IPA_00gl_80lmin\camera 1'

# Define the pixel coordinates to track (x, y) tuples
pixels_to_track = [(500, 300), (1000, 400), (300, 1000), (1100, 1000)]  # Replace with your pixel coordinates

# Initialize a dictionary to store pixel values for each frame
pixel_values = {pixel: [] for pixel in pixels_to_track}

# Iterate through all .tif files in the folder in numerical order
file_list = sorted(
    [f for f in os.listdir(folder_path) if f.endswith('.tif')],
    key=lambda x: int(x.split('_')[1].split('.')[0])  # Extract frame number from filename (e.g., img_50.tif)
)

for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    print(file_path)
    # Open the .tif image and convert to a NumPy array
    img = np.array(Image.open(file_path))
    
    # Record pixel values for each pixel in `pixels_to_track`
    for pixel in pixels_to_track:
        x, y = pixel
        pixel_values[pixel].append(img[y, x])  # Note: NumPy arrays use (row, column)

# Plot the pixel value changes over frames
plt.figure(figsize=(10, 6))
for pixel, values in pixel_values.items():
    plt.plot(values, label=f'Pixel {pixel}')

plt.title('Pixel Value Changes Across Frames')
plt.xlabel('Frame Number')
plt.ylabel('Pixel Value')
plt.legend()
plt.grid()
plt.show()



