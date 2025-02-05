from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def average_intensity(path, all_numbers):
    image_count = len(all_numbers)
    average_array = None
    # Loop through each image
    for num in all_numbers:
        image_path = path / f'img_{num}.tif'
        print(f'img_{num}.tif')
        with Image.open(image_path) as img:
            img_array = np.array(img, dtype=np.float64)  # Convert image to NumPy array
            if average_array is None:
                average_array = np.zeros_like(img_array)  # Initialize accumulator
            average_array += img_array  # Add pixel values

    # Calculate average
    average_array /= image_count

    image_array = average_array
    return image_array

def BHC(I_BH, I_empty, coefficients, mu_eff):
    x_values = np.linspace(0, 20, 2000) # range of possible distances through liquid (for interpolation)
    lnII = -np.log(I_BH/I_empty) # y value to fit

    P_x_values = np.polyval(coefficients, x_values) # predicted y values based on polynomial and x-values
    x_fit_values = np.interp(lnII, P_x_values, x_values) # use interpolation to determine ditacne through liquid based on actual data


    I_noBH = np.exp(-mu_eff * x_fit_values)* I_empty

    
    return I_noBH

def calc_coef_mu(x, y):

    coefficients = np.polyfit(x, y, 11)

    mu_eff = np.sum(x[x<x.max()/2] * y[x<x.max()/2]) / np.sum(x[x<x.max()/2]**2)
    return coefficients, mu_eff

path_to_date = Path(r'd:\XRay\2024-11-27 Rik en Sam')

# plot image with positions where beam-hardening is determined
single_image = path_to_date/"scattercorrected_IPA_00gl_0lmin" / 'camera 1' / "img_50.tif"
image = Image.open(single_image)
image_array = np.array(image)

middle_line = image_array.shape[0]//2
upper_line = image_array.shape[0]//4
lower_line = image_array.shape[0]//4 *3

plt.imshow(image_array)
plt.axhline(middle_line)
plt.axhline(upper_line)
plt.axhline(lower_line)
plt.show()

cams = [1,2,3]
compound = "IPA"
concentration = "00gl"
gasflow = "0lmin"
for i, cam in enumerate(cams):
    camera = f'camera {cam}'
    path = path_to_date / f'scattercorrected_{compound}_{concentration}_{gasflow}' / camera
    path_empty = path_to_date / f'scattercorrected_empty' / camera
    distance_cam = np.load(f'distances_cam{cam}.npy')
    print(distance_cam.max())

    all_numbers = np.arange(50,55) # change this to correct last image (no hardcoding)
    
    side_names = ['upper', 'middle', 'lower']
    lines = [upper_line, middle_line, lower_line]
    fig, ax = plt.subplots(3,2, figsize=(15,15))
    for fig_row, line in enumerate(lines):
        #row_start = row_n
    
        middle_column = distance_cam.shape[1]//2 # vertical line through middle
        left_half_x = distance_cam[line-50:line+50,:middle_column].mean(axis=0).flatten()
        right_half_x = distance_cam[line-50:line+50,middle_column:].mean(axis=0).flatten()
        
        I = average_intensity(path, all_numbers)
        I_empty = average_intensity(path_empty, all_numbers)

        I_left = I[line-50:line+50, :middle_column].mean(axis=0).flatten()
        I_right = I[line-50:line+50, middle_column:].mean(axis=0).flatten()

        I_empty_left = I_empty[line-50:line+50, :middle_column].mean(axis=0).flatten()
        I_empty_right = I_empty[line-50:line+50, middle_column:].mean(axis=0).flatten()

        ln_left = -np.log(I_left/I_empty_left)
        ln_right = -np.log(I_right/I_empty_right)

        coefficients_left, mu_eff_left = calc_coef_mu(left_half_x, ln_left)
        coefficients_right, mu_eff_right = calc_coef_mu(right_half_x, ln_right)
        mu_eff_left = 0.2
        mu_eff_right = 0.2


        I_noBH_left = BHC(I_left, I_empty_left, coefficients=coefficients_left, mu_eff=mu_eff_left)
        I_noBH_right = BHC(I_right, I_empty_right, coefficients=coefficients_right, mu_eff=mu_eff_right)

        x_fit = np.linspace(distance_cam.min(), distance_cam.max(), 1000)
        y_fit_left = np.polyval(coefficients_left, x_fit) # fit polynomial lift side
        y_fit_right = np.polyval(coefficients_right, x_fit) # fit polynomial right side

    
        ax[fig_row, 0].set_title(f'camera {cam}, {side_names[fig_row]}-left side')
        ax[fig_row, 0].scatter(left_half_x, ln_left, label='original datapoints') # plot datapoints
        ax[fig_row, 0].plot(x_fit, y_fit_left, c='r', label='polynomial fit') # plot polynomial fit
        ax[fig_row, 0].plot(x_fit, x_fit*mu_eff_left, c='k', label=f'linear fit with '+r'$\mu_{eff}$'+f'= {mu_eff_left:.2f}') # plot linear line with constant mu_eff
        ax[fig_row, 0].scatter(left_half_x, -np.log(I_noBH_left/I_empty_left), label='corrected datapoints') # plot corrected data
        ax[fig_row, 0].legend()

        ax[fig_row, 1].set_title(f'camera {cam}, {side_names[fig_row]}-right side')
        ax[fig_row, 1].scatter(right_half_x, ln_right, label='original datapoints') # plot datapoints
        ax[fig_row, 1].plot(x_fit, y_fit_right, c='r', label='polynomial fit') # plot polynomial fit
        ax[fig_row, 1].plot(x_fit, x_fit*mu_eff_right, c='k', label=f'linear fit with '+r'$\mu_{eff}$'+f'= {mu_eff_right:.2f}') # plot linear line with constant mu_e
        ax[fig_row, 1].scatter(right_half_x, -np.log(I_noBH_right/I_empty_right), label='corrected datapoints') # plot corrected data
        # ax[fig_row, 1].set_xlim(0,20)
        # ax[fig_row, 1].set_ylim(0,8)
        ax[fig_row, 1].legend()
    plt.show()


    