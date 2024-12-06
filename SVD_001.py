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