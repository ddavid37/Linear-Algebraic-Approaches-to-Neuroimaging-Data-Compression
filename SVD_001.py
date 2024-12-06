
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

plt.rcParams['figure.figsize'] = [16, 8]

# Read the DICOM file
dicom_file = pydicom.dcmread('sample_fmri_image.dcm')
A = dicom_file.pixel_array

# Convert to grayscale if necessary
if len(A.shape) == 3:
    X = np.mean(A, -1)
else:
    X = A

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

r_values = [5, 10, 20, 30, 40, 50, 75, 100]

# Save the images with different r values and calculate MSE, PSNR, and memory size
mse_values = []
psnr_values = []
memory_sizes = []

for r in r_values:
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    
    # Save the image
    plt.figure()
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title(f'r = {r}')
    filename = f'approx_image_r_{r}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Calculate MSE and PSNR
    mse = mean_squared_error(X, Xapprox)
    psnr = peak_signal_noise_ratio(X, Xapprox, data_range=X.max() - X.min())
    
    # Get memory size
    memory_size = os.path.getsize(filename)
    
    mse_values.append(mse)
    psnr_values.append(psnr)
    memory_sizes.append(memory_size)
    
    print(f'Size of {filename}: {memory_size} bytes')
    print(f'For r = {r}, MSE: {mse}, PSNR: {psnr} dB')

print('\nMSE values range from 0–∞, with lower being better.')
print('PSNR values range from 20–50 dB, with higher being better.')

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(r_values, mse_values, marker='o')
plt.xlabel('r')
plt.ylabel('MSE')
plt.title('MSE vs r')

plt.subplot(1, 3, 2)
plt.plot(r_values, psnr_values, marker='o')
plt.xlabel('r')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs r')

plt.subplot(1, 3, 3)
plt.plot(r_values, memory_sizes, marker='o')
plt.xlabel('r')
plt.ylabel('Memory Size (bytes)')
plt.title('Memory Size vs r')

plt.tight_layout()
plt.show()
