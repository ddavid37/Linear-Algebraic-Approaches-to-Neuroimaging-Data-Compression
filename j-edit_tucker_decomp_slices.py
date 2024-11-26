import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import pydicom
import matplotlib.pyplot as plt

def load_dcm_as_tensor(file_path, num_slices=69):
    # Load DICOM file
    dcm = pydicom.dcmread(file_path)
    mosaic_data = dcm.pixel_array
    print(f"Original mosaic shape: {mosaic_data.shape}")

    # Calculate grid size and slice dimensions
    grid_size = int(np.ceil(np.sqrt(num_slices)))
    slice_size = mosaic_data.shape[0] // grid_size  # Assume square grid
    tensor = np.zeros((num_slices, slice_size, slice_size))

    # Extract slices into a 3D tensor
    for i in range(num_slices):
        row = i // grid_size
        col = i % grid_size
        tensor[i] = mosaic_data[
            row * slice_size : (row + 1) * slice_size,
            col * slice_size : (col + 1) * slice_size,
        ]

    print(f"Extracted tensor shape: {tensor.shape}")
    return tensor

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
    # Load DICOM and extract slices as tensor
    file_path = "sample_fmri_image.dcm"  # Update with your actual file path
    tensor = load_dcm_as_tensor(file_path)

    # Define Tucker decomposition ranks
    ranks = [20, 20, 20]  # Adjust based on memory capacity and tensor size

    # Apply Tucker decomposition
    core, factors, reconstructed = apply_tucker_decomposition(tensor, ranks)

    # Compute metrics
    rmse, compression_ratio = calculate_metrics(tensor, reconstructed, core, factors)
    print(f"RMSE: {rmse}")
    print(f"Compression Ratio: {compression_ratio}")

    # Visualize a middle slice for original, reconstructed, and difference
    middle_slice = tensor.shape[0] // 2
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(tensor[middle_slice], cmap="gray")
    plt.title("Original")

    plt.subplot(132)
    plt.imshow(reconstructed[middle_slice], cmap="gray")
    plt.title("Reconstructed")

    plt.subplot(133)
    plt.imshow(np.abs(tensor[middle_slice] - reconstructed[middle_slice]), cmap="hot")
    plt.title("Difference")

    plt.show()