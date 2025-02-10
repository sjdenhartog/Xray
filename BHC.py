import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

all_numbers = np.arange(50,220) # all frames that are taken into account for scatter measurements
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


path = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_water_0lmin\camera 1') # column filled with water
path_empty = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_empty\camera 1') # empty column
path_scatter_empty = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_empty_scatter_S23_D1\camera 1') # scatter of empty column
path_scatter_full = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_water_0lmin_scatter_S23_D1\camera 1') # scatter of full column

path_empty2 =  Path(r'd:\XRay\2024-11-28 Rik en Sam\scattercorrected_empty\camera 1')
path2 = Path(r'd:\XRay\2024-11-28 Rik en Sam\preprocessed_AA_000gl_0lmin\camera 1')

image_array_empty = average_intensity(path_empty, all_numbers) # Intensity of empty column (without attenuation caused by the liquid in the column)
image_array_scatter_empty = average_intensity(path_scatter_empty, all_numbers)# intensit of scatter of empty column

I_empty = image_array_empty - image_array_scatter_empty

data = pd.read_csv('intensity.csv')

I_full = data['I_full']
I_full2 = average_intensity(path2, all_numbers)
#I_empty = data['I_empty']
lnII = data['-ln_intensity']
x = data['distance_liquid']

coefficients = np.polyfit(x, lnII, 7)
x_half = x[x<19/2]
mu_eff = np.sum(x_half * lnII) / np.sum(x_half**2)

#a, b, c, d = coefficients

x_fit = np.linspace(min(x), max(x), 500)
y_fit = np.polyval(coefficients, x_fit)




plt.scatter(x, lnII)
plt.plot(x_fit, y_fit, c='k', label='polyfit')
plt.plot(x, mu_eff*x, c='r', label=r'$\mu_{eff}$'+f' = {mu_eff:.2f}')

plt.xlabel('x (cm)')
plt.ylabel(r'$-ln(I(x)/I_{empty})$')
plt.legend()
plt.show()

def BHC(I_BH, I_empty, coefficients, mu_eff):
    x_values = np.linspace(0, 20, 1000) # range of possible distances through liquid (for interpolation)
    lnII = -np.log(I_BH/I_empty) # y value to fit

    P_x_values = np.polyval(coefficients, x_values) # predicted y values based on polynomial and x-values
    x_fit_values = np.interp(lnII, P_x_values, x_values) # use interpolation to determine ditacne through liquid based on actual data

    #lnII_noBH = mu_eff * x_fit_values # -ln(I_NoBH/I_empty)
    #I_noBH = -np.exp(lnII_noBH)* I_empty
    I_noBH = np.exp(-mu_eff * x_fit_values)* I_empty

    #I_noBH = None
    return I_noBH #lnII, x_fit_values

# I_BH = I_full

# lnII, x_fit_values = BHC(I_BH, I_empty, coefficients, mu_eff)

# plt.plot(x, mu_eff*x)
# plt.plot(x_fit_values, lnII, c='k')
# plt.show()

single_image = path / f'img_100.tif'
image = Image.open(single_image)
image_array = np.array(image)

# Plot the image using imshow
plt.imshow(image_array[100:600, 80:1500])
plt.show()

plt.plot(image_array[400,80:1500])
plt.show()

I_noBH = BHC(image_array, I_empty, coefficients, mu_eff=mu_eff)

# Plot the image using imshow
plt.imshow(I_noBH[100:600, 80:1500])
plt.show()

plt.plot(I_noBH[400,80:1500])
plt.show()
