import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import numpy as np
from PIL import Image
import cv2
from skimage.filters import threshold_multiotsu
from skimage.util import view_as_blocks
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def convert_tiff_to_jpg(image_path):
    if image_path.lower().endswith(('.tif', '.tiff')):
        img = Image.open(image_path)
        jpg_path = os.path.splitext(image_path)[0] + '.jpg'
        img.convert('RGB').save(jpg_path, 'JPEG')
        return jpg_path
    return image_path

optimal_cluster_number = 0
def find_optimal_clusters(data, max_clusters=10):
    # Ensure we have enough samples for clustering
    n_samples = len(data)
    if n_samples < 2:
        return 2  # Minimum number of clusters
    
    # Adjust max_clusters if needed
    max_clusters = min(max_clusters, n_samples - 1)
    if max_clusters < 2:
        return 2
    
    scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = calinski_harabasz_score(data, labels)
        scores.append(score)
    
    optimal_k = np.argmax(scores) + 2  # +2 because we started from k=2
    optimal_cluster_number = optimal_k
    return optimal_k

def adaptive_patch_size(image):
    # Ensure image is 2D
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = image.shape
    
    variances = []
    for size in [8, 16, 32, 64]:
        # Calculate padding needed
        pad_height = (size - (height % size)) % size
        pad_width = (size - (width % size)) % size
        
        # Add padding if needed
        if pad_height > 0 or pad_width > 0:
            padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='edge')
        else:
            padded_image = image
            
        # Create blocks and calculate variance
        blocks = view_as_blocks(padded_image, block_shape=(size, size))
        mean_var = np.mean(np.var(blocks, axis=(2, 3)))
        variances.append((size, mean_var))
    
    # Choose size with highest variance preservation
    return max(variances, key=lambda x: x[1])[0]

def find_optimal_patches(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate image dimensions
    height, width = image.shape
    
    # Find optimal patch size (using a simple heuristic)
    # Use 1/8th of the smaller dimension, but ensure it's a power of 2
    min_dim = min(height, width)
    patch_size = min_dim // 8
    
    # Round to nearest power of 2
    patch_size = 2 ** int(np.log2(patch_size))
    
    # Ensure patch size is between 8 and 64
    patch_size = max(8, min(64, patch_size))
    
    return patch_size



#generate a colorized image from the segmented image
# This function assigns a unique color to each cluster in the segmented image
# and returns a colorized image.
def colorize_clusters(segmented, num_clusters=optimal_cluster_number):
    """
    Assigns a unique color to each cluster in the segmented image.
    
    Parameters:
        segmented (numpy.ndarray): The segmented image with cluster labels.
        num_clusters (int): The number of clusters.
    
    Returns:
        numpy.ndarray: A colorized image with each cluster assigned a unique color.
    """
    # Create a colormap (e.g., using matplotlib or manually define colors)
    import matplotlib.pyplot as plt
    colormap = plt.cm.get_cmap('tab10', num_clusters)  # Use a colormap with `num_clusters` colors
    
    # Create an RGB image
    height, width = segmented.shape
    colorized_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for cluster_id in range(num_clusters):
        # Get the color for the current cluster
        color = (np.array(colormap(cluster_id)[:3]) * 255).astype(np.uint8)  # Convert to RGB (0-255)
        
        # Assign the color to all pixels belonging to the current cluster
        colorized_image[segmented == cluster_id] = color
    
    return colorized_image



def patch_based_kmeans(image, patch_size):
    height, width = image.shape[:2]
    patches = []
    positions = []
    
    # Extract patches
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch.flatten())
            positions.append((y, x))
    
    patches = np.array(patches)
    
    # Ensure we have enough patches for clustering
    if len(patches) < 2:
        # If not enough patches, use a simpler approach
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(patches.reshape(-1, 1))
        segmented = np.zeros_like(image)
        for (y, x), label in zip(positions, labels):
            segmented[y:y+patch_size, x:x+patch_size] = label * 255
        return segmented, 2
    
    # optimal_k = find_optimal_clusters(patches)
    
    optimal_k = 0
    # Find optimal number of clusters
    if optimal_cluster_number != 0:
        optimal_k = optimal_cluster_number
    else:
        optimal_k = find_optimal_clusters(patches)
    
    # Apply K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(patches)
    
    # Create segmented image
    segmented = np.zeros_like(image)
    for (y, x), label in zip(positions, labels):
        segmented[y:y+patch_size, x:x+patch_size] = label * (255 // (optimal_k - 1))
    
    return segmented, optimal_k

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

def patch_based_thresholding(image, patch_size):
    height, width = image.shape[:2]
    segmented = np.zeros_like(image)
    
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            threshold = np.mean(patch)
            segmented[y:y+patch_size, x:x+patch_size] = (patch > threshold) * 255
    
    return segmented

def calculate_metrics(original, segmented):
    # Calculate basic metrics (can be expanded)
    mse = np.mean((original - segmented) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return mse, psnr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process_with_custom_patch', methods=['POST'])
def process_with_custom_patch():
    try:
        filename = request.form['filename']
        custom_patch_size = int(request.form['patch_size'])
        cluster_no = int(request.form['cluster_no'])
        
        # Read the original image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to read the image file'}), 400
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Process with custom patch size and cluster number
        optimal_cluster_number = cluster_no
        kmeans_result, optimal_k = patch_based_kmeans(image, custom_patch_size)
        multiotsu_result = patch_based_multiotsu(image, custom_patch_size)
        threshold_result = patch_based_thresholding(image, custom_patch_size)
        
        colorized_kmeans_result = colorize_clusters(kmeans_result, optimal_k)
        colorized_multiotsu_result = colorize_clusters(multiotsu_result, min(optimal_cluster_number, 5))
        
        # Save results
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], 'kmeans_result.jpg'), colorized_kmeans_result)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], 'multiotsu_result.jpg'), colorized_multiotsu_result)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], 'threshold_result.jpg'), threshold_result)
        
        # Calculate metrics
        kmeans_metrics = calculate_metrics(image, kmeans_result)
        multiotsu_metrics = calculate_metrics(image, multiotsu_result)
        threshold_metrics = calculate_metrics(image, threshold_result)
        
        # Convert metrics to Python native types
        kmeans_metrics = [float(x) for x in kmeans_metrics]
        multiotsu_metrics = [float(x) for x in multiotsu_metrics]
        threshold_metrics = [float(x) for x in threshold_metrics]
        
        # Determine best method
        metrics = {
            'K-means': kmeans_metrics[1],
            'Multi-Otsu': multiotsu_metrics[1],
            'Thresholding': threshold_metrics[1]
        }
        best_method = max(metrics.items(), key=lambda x: x[1])
        
        return jsonify({
            'optimal_k': int(optimal_cluster_number),  # Convert numpy.int64 to Python int
            'kmeans_metrics': kmeans_metrics,
            'multiotsu_metrics': multiotsu_metrics,
            'threshold_metrics': threshold_metrics,
            'best_method': best_method[0],
            'best_psnr': float(best_method[1])  # Convert numpy.float64 to Python float
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return render_template('error.html', message='No file part in the request'), 400
        
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', message='No file selected'), 400
        
        if not file or not allowed_file(file.filename):
            return render_template('error.html', message='Invalid file type. Please upload a PNG, JPG, JPEG, TIFF, or TIF file.'), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert TIFF to JPG if necessary
        filepath = convert_tiff_to_jpg(filepath)
        if filepath.endswith('.jpg'):
            filename = os.path.basename(filepath)
        
        # Read and process image
        image = cv2.imread(filepath)
        if image is None:
            return render_template('error.html', message='Failed to read the image file'), 400
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find optimal patch size
        patch_size = find_optimal_patches(image)
        
        # Apply different segmentation methods
        kmeans_result, optimal_k = patch_based_kmeans(image, patch_size)
        multiotsu_result = patch_based_multiotsu(image, patch_size)
        threshold_result = patch_based_thresholding(image, patch_size)

        # Colorize the segmented image
        colorized_kmeans_result = colorize_clusters(kmeans_result, optimal_k)
        colorized_multiotsu_result = colorize_clusters(multiotsu_result, min(optimal_cluster_number, 5))
        
        # Save results
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], 'kmeans_result.jpg'), colorized_kmeans_result)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], 'multiotsu_result.jpg'), colorized_multiotsu_result)
        cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], 'threshold_result.jpg'), threshold_result)
        
        # Calculate metrics
        kmeans_metrics = calculate_metrics(image, kmeans_result)
        multiotsu_metrics = calculate_metrics(image, multiotsu_result)
        threshold_metrics = calculate_metrics(image, threshold_result)
        
        # Determine best method
        metrics = {
            'K-means': kmeans_metrics[1],
            'Multi-Otsu': multiotsu_metrics[1],
            'Thresholding': threshold_metrics[1]
        }
        best_method = max(metrics.items(), key=lambda x: x[1])
        
        return render_template('results.html',
                             original_image=filename,
                             kmeans_image='kmeans_result.jpg',
                             multiotsu_image='multiotsu_result.jpg',
                             threshold_image='threshold_result.jpg',
                             patch_size=patch_size,
                             optimal_k=optimal_k,
                             kmeans_metrics=kmeans_metrics,
                             multiotsu_metrics=multiotsu_metrics,
                             threshold_metrics=threshold_metrics,
                             best_method=best_method[0],
                             best_psnr=best_method[1],
                             cluster_no=optimal_cluster_number)
    except Exception as e:
        return render_template('error.html', message=f'An error occurred: {str(e)}'), 500

if __name__ == '__main__':
    #app.run(debug=True) 
    app.run(host='0.0.0.0', port=5000, debug=True)