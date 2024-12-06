import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom

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

U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 10, 20, 30, 40, 50, 75, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()
	

import os

# Save the images with different r values
for r in (5, 10, 20, 30, 40, 50, 75, 100):
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.figure()
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    filename = f'approx_image_r_{r}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f'Size of {filename}: {os.path.getsize(filename)} bytes')

from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

for r in (5, 10, 20, 30, 40, 50, 75, 100):
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    mse = mean_squared_error(X, Xapprox)
    psnr = peak_signal_noise_ratio(X, Xapprox, data_range=X.max() - X.min())
    print(f'For r = {r}, MSE: {mse}, PSNR: {psnr} dB')

print('\nMSE values range from 0–∞, with lower being better.')
print('PSNR values range from 20–50 dB, with higher being better.')
