import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import pydicom
import matplotlib.pyplot as plt

def load_dcm_as_3d_tensor(file_path, num_slices=69):
    # Load DICOM file
    dcm = pydicom.dcmread(file_path)
    mosaic_data = dcm.pixel_array
    print(f"Original mosaic shape: {mosaic_data.shape}")
    
    # Calculate grid size and slice dimensions
    grid_size = int(np.ceil(np.sqrt(num_slices)))  # Assume square grid
    slice_size = mosaic_data.shape[0] // grid_size  # Size of each slice

    # Reshape the mosaic into a 3D tensor
    reshaped_data = mosaic_data.reshape(grid_size, slice_size, grid_size, slice_size)
    tensor_3d = reshaped_data.transpose(0, 2, 1, 3).reshape(num_slices, slice_size, slice_size)
    print(f"Reshaped to 3D tensor with shape: {tensor_3d.shape}")
    return tensor_3d

def apply_tucker_decomposition(tensor, ranks):
    # Convert to tensorly tensor
    tl_tensor = tl.tensor(tensor)
    print(f"Tensor shape before decomposition: {tl_tensor.shape}")

    # Perform Tucker decomposition
    core, factors = tucker(tl_tensor, rank=ranks)
    reconstructed = tl.tucker_to_tensor((core, factors))
    return core, factors, reconstructed

def calculate_metrics(original, reconstructed, core, factors):
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    original_size = np.prod(original.shape)
    compressed_size = np.prod(core.shape) + sum(f.size for f in factors)
    compression_ratio = original_size / compressed_size
    return rmse, compression_ratio

if __name__ == "__main__":
    # Load DICOM as 3D tensor
    file_path = "sample_fmri_image.dcm"  # Update according to filename in-use
    tensor_3d = load_dcm_as_3d_tensor(file_path)

    # Define Tucker decomposition ranks 
    # (each rank corresponds to [number of slices, width dimension, height dimension])
    ranks = [20, 50, 50]  # Adjust ranks based on memory and compression needs

    # Apply Tucker decomposition
    core, factors, reconstructed = apply_tucker_decomposition(tensor_3d, ranks)

    # Compute metrics
    rmse, compression_ratio = calculate_metrics(tensor_3d, reconstructed, core, factors)
    print(f"RMSE: {rmse}")
    print(f"Compression Ratio: {compression_ratio}")

    # Visualize the middle slice of the original, reconstructed, and difference
    middle_slice = tensor_3d.shape[0] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(tensor_3d[middle_slice], cmap="gray")
    plt.title("Original Slice")

    plt.subplot(132)
    plt.imshow(reconstructed[middle_slice], cmap="gray")
    plt.title("Reconstructed Slice")

    plt.subplot(133)
    plt.imshow(np.abs(tensor_3d[middle_slice] - reconstructed[middle_slice]), cmap="hot")
    plt.title("Difference")

    plt.show()