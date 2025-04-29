import scipy.io
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu

# Load the .mat file
mat = scipy.io.loadmat('Indian_pines.mat')

# Extract the image data (replace 'indian_pines' with the correct key if different)
image_data = mat['indian_pines']

# Reshape the data to 2D (pixels x bands) for PCA
pixels, bands = image_data.shape[0] * image_data.shape[1], image_data.shape[2]
image_data_reshaped = image_data.reshape(pixels, bands)

# Apply PCA to reduce to 1 component
pca = PCA(n_components=1)
image_data_pca = pca.fit_transform(image_data_reshaped)

# Reshape back to 2D (original spatial dimensions)
image_data_pca_reshaped = image_data_pca.reshape(image_data.shape[0], image_data.shape[1])

# Normalize the data to fit into the 8-bit range (0-255)
image_data_normalized = (255 * (image_data_pca_reshaped - np.min(image_data_pca_reshaped)) / 
                         (np.max(image_data_pca_reshaped) - np.min(image_data_pca_reshaped))).astype(np.uint8)

# Convert to PIL Image
image = Image.fromarray(image_data_normalized)

# Save as TIFF
# image.save('output_pca.tiff')
# print("TIFF image saved as 'output_pca.tiff'")

def patch_based_multiotsu(image, patch_size):
    height, width = image.shape[:2]
    segmented = np.zeros_like(image)
    
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            try:
                # Increase number of bins and add error handling
                thresholds = threshold_multiotsu(patch, nbins=256, classes=5)#min(optimal_cluster_number, 5))
                regions = np.digitize(patch, bins=thresholds)
                segmented[y:y+patch_size, x:x+patch_size] = regions * 85
            except ValueError:
                # If multi-otsu fails, fall back to simple thresholding
                threshold = np.mean(patch)
                segmented[y:y+patch_size, x:x+patch_size] = (patch > threshold) * 170
    
    return segmented

print(patch_based_multiotsu(image_data_normalized, 145))

