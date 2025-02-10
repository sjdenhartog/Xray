from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_BHC = r'd:\XRay\2024-11-27 Rik en Sam\BHC2_IPA_00gl_0lmin\camera 1\img_100.tif'
img_scattercorrected =  r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_IPA_00gl_0lmin\camera 1\img_100.tif'
img_preprocessed = r'd:\XRay\2024-11-27 Rik en Sam\preprocessed_IPA_00gl_0lmin\camera 1\img_100.tif'

img_name = ['preprocessed', 'scattercorrected', 'BHC']

image_paths = [
    img_preprocessed,
    img_scattercorrected,
    img_BHC,

]
# Read images and convert them to NumPy arrays
images = [np.array(Image.open(path)) for path in image_paths]

# Determine the global min and max pixel values across all images
vmin = min(img.min() for img in images)  # Find the minimum value across all images
vmax = 16000#max(img.max() for img in images)  # Find the maximum value across all images

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (ax, img) in enumerate(zip(axes, images)):
    im = ax.imshow(img, vmin=vmin, vmax=vmax)  # Apply global vmin and vmax
    ax.set_title(img_name[i])
    ax.axis('off')  # Hide axes for better visualization

# Add a shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, pad=0.05)
cbar.set_label('Pixel Intensity')

# Show the plot
plt.tight_layout()
plt.show()

path_to_image = r'd:\XRay\2024-11-27 Rik en Sam\BHC2_IPA_00gl_80lmin\camera 1\img_100.tif'
#path_to_image = r'D:\XRay\2024-11-21 Rik en Sam\preprocessed_PrOH_00gl_80lmin\camera 1\img_100.tif'

image = Image.open(path_to_image)
image_array = np.array(image)

fig, ax = plt.subplots(2, 1,figsize=(15,15))
ax[0].imshow(image_array)
ax[1].plot(image_array[600, 200:1400])
plt.show()

