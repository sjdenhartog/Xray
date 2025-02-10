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

path_to_date = Path(r'd:\XRay\2024-11-27 Rik en Sam')



path_empty1 =  Path(r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_empty\camera 1')
path1 = Path(r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_IPA_00gl_0lmin\camera 1')
distance_cam1 = np.load('distances_cam1.npy')

path_empty2 =  Path(r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_empty\camera 2')
path2 = Path(r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_IPA_00gl_0lmin\camera 2')
distance_cam2 = np.load('distances_cam2.npy')

path_empty3 =  Path(r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_empty\camera 3')
path3 = Path(r'd:\XRay\2024-11-27 Rik en Sam\scattercorrected_IPA_00gl_0lmin\camera 3')
distance_cam3 = np.load('distances_cam3.npy')

cams = [1,2,3]
compound = "IPA"
concentration = "00gl"
gasflow = "0lmin"
for i, cam in cams:
    camera = f'camera {cam}'
    path = path_to_date / f'scattercorrected_{compound}_{concentration}_{gasflow}'
    path_empty = path_to_date / f'scattercorrected_empty' / camera
    distance_cam = np.load(f'distance_cam{cam}.npy')

    all_numbers = np.arange(50,220) # change this to correct last image (no hardcoding)
    



def calc_coef_mu(path, path_empty, distance):

    all_numbers = np.arange(50,220)
    I_full = average_intensity(path, all_numbers)
    I_empty = average_intensity(path_empty, all_numbers)

    y = -np.log((I_full[550:650, 60:1500].mean(axis=0).flatten())/(I_empty[550:650, 60:1500].mean(axis=0).flatten()))
    x = distance[550:650, 60:1500].mean(axis=0).flatten()

    coefficients = np.polyfit(x, y, 3)

    mu_eff = np.sum(x * y) / np.sum(x**2)
    return coefficients, mu_eff

all_full = [path1, path2, path3]
all_empty = [path_empty1, path_empty2, path_empty3]
all_distance = [distance_cam1, distance_cam2, distance_cam3]


coefficients_all = []
mu_eff_all = []
for i in range(3):

    coefficients, mu_eff = calc_coef_mu(all_full[i], all_empty[i], all_distance[i])
    coefficients_all.append([coefficients])
    mu_eff_all.append(mu_eff)


def BHC(I_BH, I_empty, coefficients, mu_eff):
    x_values = np.linspace(0, 20, 1000) # range of possible distances through liquid (for interpolation)
    lnII = -np.log(I_BH/I_empty) # y value to fit

    P_x_values = np.polyval(coefficients, x_values) # predicted y values based on polynomial and x-values
    x_fit_values = np.interp(lnII, P_x_values, x_values) # use interpolation to determine ditacne through liquid based on actual data


    I_noBH = np.exp(-mu_eff * x_fit_values)* I_empty

    
    return I_noBH 

for folder in path_to_date.iterdir():
    if folder.is_dir(): 
        if folder.name.startswith('scattercorrected_'): 
            for cam in range(1,4):
                path_to_scatcor_cam = Path(folder.name) / f'camera {cam}'
                print(path_to_scatcor_cam)

I_noBH = BHC(I_BH= I_full, I_empty=I_empty, coefficients=coefficients, mu_eff=mu_eff)
y_noBH = -np.log((I_noBH[550:650, 60:1500].mean(axis=0).flatten())/(I_empty[550:650, 60:1500].mean(axis=0).flatten()))

plt.scatter(x, y, label='with beam-hardening')
plt.scatter(x, y_noBH, label='no beam-hardening')
plt.plot(np.linspace(0, 19, 1000), np.polyval(coefficients, np.linspace(0, 19, 1000)), label='poly-fit')
plt.plot(x, mu_eff*x, label='with mu_eff')
plt.legend()
plt.show()




# def scattercorrected_output_dir(path, camera, gasflow):
#     if gasflow == 0:
#         # for 0lmin
#         path_to_prepro_folder_0lmin = re.sub(r'_scatter.*', '', str(path)) 

#         # Convert back to Path object
#         path_to_prepro_folder_0lmin = Path(path_to_prepro_folder_0lmin)

#         path_to_prepro_cam = path_to_prepro_folder_0lmin / camera
        
#         output_dir = re.sub(r'preprocessed.', 'scattercorrected_', str(path_to_prepro_cam))

#  last_file = len(list(path_to_prepro_cam_0.glob("*.tif")))
