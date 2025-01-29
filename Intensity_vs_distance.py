#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:16:08 2024

@author: samdenhartog
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy.optimize import curve_fit, minimize


def cate_to_astra(path, det, geom_scaling_factor=None, angles=None):
    """Convert `Geometry` objects from our calibration package to the
    ASTRA vector convention."""

    import pickle
    from cate import astra, xray
    from numpy.lib.format import read_magic, _check_version, _read_array_header

    class RenamingUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "StaticGeometry":
                name = "Geometry"
            return super().find_class(module, name)

    with open(path, "rb") as fp:
        version = read_magic(fp)
        _check_version(version)
        dtype = _read_array_header(fp, version)[2]
        assert dtype.hasobject
        multicam_geom = RenamingUnpickler(fp).load()[0]

    detector = astra.Detector(
        det["rows"], det["cols"], det["pixel_width"], det["pixel_height"]
    )

    def _to_astra_vec(g):
        v = astra.geom2astravec(g, detector.todict())
        if geom_scaling_factor is not None:
            v = np.array(v) * geom_scaling_factor
        return v

    if angles is None:
        geoms = []
        for _, g in sorted(multicam_geom.items()):
            geoms.append(_to_astra_vec(g))
        return geoms
    else:
        geoms_all_cams = {}
        for cam in list(multicam_geom.keys()):
            geoms = []
            for a in angles:
                g = xray.transform(multicam_geom[cam], yaw=a)
                geoms.append(_to_astra_vec(g))
            geoms_all_cams[cam] = geoms

        return geoms_all_cams

def compute_all_pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols):
    # Create a 2D grid of row indices (i) and column indices (j)
    i_vals = np.arange(-rows // 2, rows // 2).reshape(-1, 1) # Shape (rows, 1) the middle of the detector is at (0,0), so substract half the number of rows
    j_vals = np.arange(-cols // 2, cols // 2).reshape(1, -1) # Shape (1, cols) the middle of the detector is at (0,0), so substract half the number of cols
    print(i_vals)
    
    # Compute the 3D coordinates for all pixels in centimeters
    x = (dX + j_vals * uX + i_vals * vX) # coordinate of the detector middle corrected for every pixel based on vector describing distance between pixels
    y = (dY + j_vals* uY + i_vals * vY)
    z = (dZ + j_vals * uZ + i_vals* vZ)
    
    return x, y, z


# data = np.load('/Users/samdenhartog/Documents/LST/Master/Year 2/MEP/code/Geometry/geom_preprocessed_Needles_Rotation_5degps_calibrated_on_04jul2024.npy', allow_pickle=True)

# path to geometry
path = Path(r'D:\XRay\2024-11-14 Rik en Sam\preprocessed_Rotation_needles_5degps_again\calibration\geom_preprocessed_Rotation_needles_5degps_again_calibrated_on_14janc2025.npy')

det = {
    "rows": 1524,        # Number of rows in the detector
    "cols": 1548,        # Number of columns in the detector
    "pixel_width": 0.0198,  # Pixel width in cm
    "pixel_height": 0.0198,  # Pixel height in cm
    'det_width': 30.7, # cm, detector width
    'det_height': 30.2, # cm, detector height
    'column_inner_D' : 19, # cm
    'column_outer_D' : 20.0 # cm
    
}


geoms_all_cams = cate_to_astra(path=path, det=det)


def calc_distance_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='inner'):
    # Directional vector is build up of a, b and c 
    a = x_coords - srcX # in x direction
    b = y_coords - srcY # in y direction
    c = z_coords - srcZ # in z direction

    # Equations for all the straight lines connecting the X-ray source and the pixel on the detector
    # x = srcX + a*t
    # y = srcY + b*t
    # z = srcZ + c*t
    #
    # Equation for cylinder:
    # x**2 + y**2 = r**2
    if diameter_type == 'inner':
        D = det['column_inner_D']# / det['pixel_width']
    elif diameter_type == 'outer':
        D = det['column_outer_D']

    r = D/2 # cm


    # Coefficients for the quadratic equation (see notes: write equation as quadratic equation for t. so A*t**2 + B*t + C = 0)
    A = a**2 + b**2
    B = 2 * (srcX * a + srcY * b)
    C = srcX**2 + srcY**2 - r**2

    # discriminant
    Disc = B**2 - 4 * A * C

    # Handle cases where the discriminant is negative (no real solution)
    Disc[Disc < 0] = 0 #np.nan # these lines do not intercept the cylinder

    print(np.count_nonzero(np.isnan(Disc)))

    # Two possible solutions for t
    t1 = (-B + np.sqrt(Disc)) / (2 * A)
    t2 = (-B - np.sqrt(Disc)) / (2 * A)

    # coordinates of first intercept through cylinder
    x_A = srcX + a * t1
    y_A = srcY + b * t1
    z_A = srcZ + c * t1

    # Coordinates of second intercept through cylinder
    x_B = srcX + a * t2
    y_B = srcY + b * t2
    z_B = srcZ + c * t2

    # Calculate the distance through the column
    d = np.sqrt((x_B - x_A)**2+(y_B-y_A)**2+(z_B-z_A)**2)

    return d

( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ ) = geoms_all_cams[0] # for camera 1?????? coordinates of source (src), coordinates of detector (d), 
                                                                             # the vector from detector pixel (0,0) to (0,1) (u),  the vector from detector pixel (0,0) to (1,0) (v)

rows, cols = det['rows'], det['cols']
# print(f'dX = {dX}')
# print(f'dY = {dY}')
# print(f'dZ = {dZ}')

# Compute the 3D coordinates for all pixels
x_coords, y_coords, z_coords = compute_all_pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols)

d = calc_distance_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det) # distance through water (inner diameter) for all pixels
d_outer = calc_distance_through_column(x_coords, y_coords, z_coords, srcX, srcY, srcZ, det, diameter_type='outer') # distance through water+column (outer diameter) for all pixels
all_numbers = np.arange(50,220) # all frames that are taken into account
# Initialize variables

image_count = len(all_numbers)
path = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_water_0lmin\camera 1') # column filled with water

def average_intensity(path, all_numbers):
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
image_array = average_intensity(path, all_numbers)

path_empty = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_empty\camera 1') # empty column
path_scatter_empty = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_empty_scatter_S23_D1\camera 1') # scatter of empty column
path_scatter_full = Path(r'D:\XRay\2024-11-19 Rik en Sam\preprocessed_water_0lmin_scatter_S23_D1\camera 1') # scatter of full column

image_array_empty = average_intensity(path_empty, all_numbers) # Intensity of empty column (without attenuation caused by the liquid in the column)
image_array_scatter_empty = average_intensity(path_scatter_empty, all_numbers)# intensit of scatter of empty column
image_array_scatter_full = average_intensity(path_scatter_full, all_numbers) # intensity of scatter of full column



row_start = 550
row_end = 650

col_start = 60#120 #60
col_end = 1500#1000 # 1500



#distance = d[row_start:row_end,col_start:col_end].flatten()
mean_distance = d[row_start:row_end,col_start:col_end].mean(axis=0).flatten() # distance through water (inner diameter)

mean_distance_outer = d_outer[row_start:row_end,col_start:col_end].mean(axis=0).flatten() # through water+column (outer diameter)

distance_through_column = mean_distance_outer - mean_distance

ratio_column_water = distance_through_column / mean_distance_outer
# ratio_column_water = np.divide(
#     distance_through_column, 
#     mean_distance, 
#     out=np.full_like(distance_through_column, np.nan),  # Default to NaN where division is invalid
#     where=mean_distance != 0
# )
print(ratio_column_water.mean())

intensity_scatter_empty = image_array_scatter_empty[row_start:row_end,col_start:col_end].mean(axis=0).flatten()
intensity_scatter_full = image_array_scatter_full[row_start:row_end,col_start:col_end].mean(axis=0).flatten()

#I_empty = image_array_empty[row_start:row_end,col_start:col_end].flatten() - image_array_scatter_empty[row_start:row_end,col_start:col_end].flatten() # correct for scatter
I_empty = image_array_empty[row_start:row_end,col_start:col_end].mean(axis=0).flatten() - intensity_scatter_empty # correct for scatter 
I_full = image_array[row_start:row_end,col_start:col_end].mean(axis=0).flatten() - intensity_scatter_full

#lnII = -np.log((image_array[row_start:row_end,col_start:col_end].flatten()- image_array_scatter_full[row_start:row_end,col_start:col_end].flatten())/I_empty) # I0 is I_empty
ln_intensity = -np.log(I_full/I_empty) # I0 is I_empty
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].plot(mean_distance, ln_intensity, '--o', alpha=0.5)#, alpha=0.5, s=5)
ax[0].set_xlabel('x (cm)', fontsize=15)
ax[0].set_ylabel(r'$-ln(\frac{I(x)}{I_{empty}})$', fontsize=15)


# mu = np.dot(distance, lnII) / np.dot(distance, distance)

# y_pred = mu * distance
# #plt.plot(distance, y_pred)


# def baur(x, a, b, c):
#     epsilon = 1e-10  # Small constant to avoid division by zero
#     mu = a + b / (x**c + epsilon)
#     lnII = mu * x
#     return lnII

# def sse(params, x, y):
#     a, b, c = params
#     y_pred = baur(x, a, b, c)
#     return np.sum((y - y_pred) ** 2)

# initial_guess = [-0.8, 1.2, 0.1]

# # Minimize the SSE
# result = minimize(sse, initial_guess, args=(distance, lnII), method='Nelder-Mead')

# # Extract the fitted parameters
# a_fit, b_fit, c_fit = result.x
# print(f'a = {a_fit}, b = {b_fit}, alpha = {c_fit}')

# # popt, pcov = curve_fit(baur, distance, lnII, p0=[-0.8, 1.2, 0.1])
# # a_fit, b_fit, c_fit = popt
# all_x = np.linspace(0.1,20, 100)
# all_y = baur(all_x, a_fit, b_fit, c_fit)

# #ax[0].plot(all_x, all_y)
# print(single_height_distance)
#ax[1].plot(all_x, all_y/all_x)
ax[1].plot(mean_distance, ln_intensity/mean_distance, '--o', alpha=0.5)
ax[1].set_xlabel('x (cm)', fontsize=15)
ax[1].set_ylabel(r'$\mu_{eff}$'+' (1/cm)', fontsize=15)

plt.tight_layout()
plt.show()

plt.plot(mean_distance, ratio_column_water)
plt.ylabel(r'$x_{wall}/x_{liquid}$')
plt.xlabel(r'$x_{liquid}$' + '(cm)')
plt.show()
# plt.plot(image_array[700, :])
# plt.show()
# d
def plot_full_geom(geom_all_cams):
    num_cam = len(geom_all_cams) # number of cameras
    fig, ax = plt.subplots()

    circle = plt.Circle((0,0), det['column_outer_D']/2, fill=False)
    ax.add_patch(circle)
    for i in range(num_cam):

        (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ ) = geoms_all_cams[i]

        x_coords, y_coords, z_coords = compute_all_pixel_coordinates(dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, rows, cols)

        
        ax.scatter([srcX, dX], [srcY, dY], label=[f'source {i+1}', f'detector {i+1}'])
        ax.scatter(x_coords, y_coords)


        #ax.set_xlim(-80, 80)
        #ax.set_ylim(-50,120)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Geometry of system')
    plt.show()
    return print('Plotted geometry of system with succes')

#########################################################################

## To plot the geometry of the system    
#plot_full_geom(geoms_all_cams)

#########################################################################


# Define the value range
min_value = 16000
max_value = 20000

col_start = 60
col_end = 1500

# Find the locations where the values fall within the range
#points = np.where((image_array[200:600,100:1500] >= min_value) & (image_array[200:600,100:1500] <= max_value))
points = np.where((image_array[row_start:row_end,col_start:col_end] >= min_value) & (image_array[row_start:row_end,col_start:col_end] <= max_value))
#no_column = np.where((d[200:600,100:1500]==0))# & (image_array[200:600,100:1500] >= min_value) & (image_array[200:600,100:1500] <= max_value))
no_column = np.where((d[row_start:row_end,col_start:col_end]==0))

plt.imshow(image_array[row_start:row_end,col_start:col_end])#, cmap='gray')

# Overlay red dots on the points where the values are between 2500 and 6000
plt.scatter(points[1], points[0], color='red', marker='o', s=5)  # Red dots for values in range
plt.scatter(no_column[1], no_column[0], color='blue', marker='o', s=5, alpha=0.1)



plt.show()

print(image_array.shape); print(d.shape)
print(image_array[200,0:10]); print(d[0,0:5])
print(image_array[50,10])

plt.imshow(image_array)
plt.show()


