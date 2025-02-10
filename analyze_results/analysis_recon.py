import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Open the HDF5 file in read mode
file_path = r"D:\XRay\2024-11-27 Rik en Sam\scattercorrected_IPA_00gl_80lmin\recon.hdf5"  # Replace with the path to your file
#file_path = r'D:\XRay\2024-11-21 Rik en Sam\preprocessed_PrOH_00gl_80lmin\recon.hdf5'
#file_path = Path(file_path)
with h5py.File(file_path, 'r') as hdf:
    # List all groups and datasets in the file
    print("Keys in the file:")
    for key in hdf.keys():
        print(key)

    # Access a specific group or dataset
    dataset = hdf['reconstruction']  # Replace with a dataset key from the file
    print("\nDataset details:")
    print(f"Shape: {dataset.shape}")
    print(f"Data type: {dataset.dtype}")
    print("\nData:")
    print(dataset[:])  # Load the dataset into a NumPy array


    slice_data = dataset[:, :, 250]  # Example slice
    print(f"Slice shape: {slice_data.shape}")
    # Compute a threshold to clip high-intensity values (e.g., 99th percentile)
    threshold = np.percentile(slice_data, 100)  # 99th percentile
    print(f"Clipping threshold: {threshold}")

    # Clip values above the threshold
    clipped_data = np.clip(slice_data, slice_data.min(), threshold)

    
    clipped_data = clipped_data.astype(np.float32)
    
    
    plt.imshow(clipped_data)#, cmap='gray')
    plt.colorbar()  # Optional: add a colorbar
    #plt.title("Clipped and Normalized Slice")
    plt.show()