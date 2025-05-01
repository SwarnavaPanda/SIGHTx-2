from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import cv2

# [Keep all your existing helper functions here (preview_images, pad_image_to_tile_multiple, etc.)]


def is_jupyter():
    """Detect if running in Jupyter notebook or Colab"""
    try:
        # Check for IPython kernel
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # type: ignore
            return False
    except (ImportError, AttributeError):
        return False
    return True

def preview_images(*images, titles=None, figsize=(15, 5)):
    """
    Display multiple images side by side for comparison using matplotlib.
    Adapts display method based on execution environment.
    """
    n_images = len(images)
    if n_images == 0:
        print("No images provided")
        return

    # Create figure and subplots
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]

    # Set default titles
    titles = titles or [f'Image {i+1}' for i in range(n_images)]

    for i, (img, title) in enumerate(zip(images, titles)):
        # Image loading and processing (keep your existing code here)
        # [Your existing image loading/processing code]

        # Display image
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()

    # Environment-appropriate display
    if is_jupyter():
        # Inline display for Jupyter/Colab
        plt.show()
    else:
        # Interactive window for terminal
        plt.show(block=True)

def pad_image_to_tile_multiple(img, tile_size=(64, 64), overlap=0):  # Add overlap parameter
    h, w = img.shape
    th, tw = tile_size
    
    # Calculate padding needed for splitting with overlap
    pad_h = (th - (h % (th - overlap)) % (th - overlap)) if (th - overlap) != 0 else 0
    pad_w = (tw - (w % (tw - overlap)) % (tw - overlap)) if (tw - overlap) != 0 else 0
    
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded_img, pad_h, pad_w

def split_into_tiles(img, tile_size=(64, 64), overlap=0):
    h, w = img.shape
    th, tw = tile_size
    step_y = th - overlap
    step_x = tw - overlap

    tiles = []
    positions = []

    # Calculate valid range considering overlap
    y_range = range(0, h - th + step_y, step_y)
    x_range = range(0, w - tw + step_x, step_x)
    
    for y in y_range:
        for x in x_range:
            tiles.append(img[y:y+th, x:x+tw])
            positions.append((y, x))
    
    return tiles, positions

def merge_tiles_with_overlap(tiles, image_shape, tile_size, overlap):
    height, width = image_shape
    tile_h, tile_w = tile_size
    step_h = tile_h - overlap
    step_w = tile_w - overlap

    merged = np.zeros(image_shape, dtype=np.float32)
    count = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for y in range(0, height - overlap, step_h):
        for x in range(0, width - overlap, step_w):
            tile = tiles[idx].astype(np.float32)
            h, w = tile.shape

            end_y = min(y + h, height)
            end_x = min(x + w, width)
            tile_h_adj = end_y - y
            tile_w_adj = end_x - x

            merged[y:end_y, x:end_x] += tile[:tile_h_adj, :tile_w_adj]
            count[y:end_y, x:end_x] += 1
            idx += 1

    count[count == 0] = 1  # Prevent division by zero
    merged_avg = merged / count
    merged_avg = np.clip(merged_avg, 0, 255).astype(np.uint8)

    return merged_avg

def canny_with_otsu(tile):
    # Compute Otsu's threshold
    otsu_thresh, _ = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thresh = int(otsu_thresh)
    low_thresh = int(0.4 * high_thresh)
    
    # Apply Canny edge detection
    edges = cv2.Canny(tile, low_thresh, high_thresh)
    return edges

def remove_small_islands(bin_img, max_size=2):
    """
    Remove white islands of area ≤ max_size, and
    black islands (holes) of area ≤ max_size, in a binary image.
    
    bin_img : uint8 array with values {0,255}
    max_size: maximum pixel area for an island to remove
    """
    # 1) Remove small white islands
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    cleaned = bin_img.copy()
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_size:
            cleaned[labels == label] = 0   # paint tiny white blob to black
    
    # 2) Remove small black islands (holes)
    inv = cv2.bitwise_not(cleaned)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_size:
            cleaned[labels == label] = 255  # paint tiny black hole to white
    
    return cleaned

def evaluate_segmentation(pred_labels, gt_labels, iou_threshold=0.5):
    """
    Evaluate multi-instance segmentation by matching predicted clusters to ground-truth clusters.

    Parameters
    ----------
    pred_labels : 2D array of ints
        Predicted label image (each cluster has a unique integer label; 0 assumed background).
    gt_labels : 2D array of ints
        Ground-truth label image (same format).
    iou_threshold : float
        Minimum IoU for a match to count as a true positive.

    Returns
    -------
    metrics : dict
        {
          'tp': int,   # true positives
          'fp': int,   # false positives
          'fn': int,   # false negatives
          'precision': float,
          'recall': float,
          'f1_score': float,
          'ious':   2D array of IoUs (pred vs gt)
        }
    """
    # 1) List of non-background labels
    pred_ids = np.setdiff1d(np.unique(pred_labels), [0])
    gt_ids   = np.setdiff1d(np.unique(gt_labels),   [0])

    # 2) Build IoU matrix
    ious = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        pred_mask = (pred_labels == pid)
        for j, gid in enumerate(gt_ids):
            gt_mask = (gt_labels == gid)
            inter = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or (pred_mask, gt_mask).sum()
            ious[i, j] = inter / union if union > 0 else 0.0

    # 3) For each predicted, find best GT; mark matches above threshold
    matched_pred = set()
    matched_gt   = set()
    for i, pid in enumerate(pred_ids):
        # best GT for this pred
        j_best = np.argmax(ious[i])
        if ious[i, j_best] >= iou_threshold:
            matched_pred.add(pid)
            matched_gt.add(gt_ids[j_best])

    # 4) Count TP, FP, FN
    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids)   - tp

    # 5) Compute precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ious': ious
    }

# Map each label to a random color for visualization
def visualize_labels(label_img):
    label_hue = np.uint8(179 * label_img / np.max(label_img))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_img == 0] = 0  # Set background to black
    return labeled_img





def segmentation_pipeline(
    image_path, 
    gt_mat_path,
    tile_size=(64, 64),
    overlap=16,
    # Keep all original parameters but make them optional
    denoise_params=None,
    morph_kernels=None,
    min_area=7,
    extent_thresh=0.2,
    island_size=15,
    iou_threshold=0.1,
    visualize=False,
    **kwargs  # For forward compatibility
):
    """Complete segmentation pipeline with parameterized components."""
    

    denoise_params = denoise_params or {'d': 0, 'sigmaColor': 20, 'sigmaSpace': 75}
    morph_kernels = morph_kernels or [(2,2), (3,3)]
    # Load and preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_denoised = cv2.bilateralFilter(img, **denoise_params)
    
    # Tile processing
    img_padded, pad_h, pad_w = pad_image_to_tile_multiple(img_denoised, tile_size, overlap)
    tiles, positions = split_into_tiles(img_padded, tile_size, overlap)
    edge_tiles = [canny_with_otsu(tile) for tile in tiles]
    
    # Merge and post-process
    merged = merge_tiles_with_overlap(edge_tiles, img_padded.shape, tile_size, overlap)
    merged_bin = np.where(merged > 0.1, 255, 0).astype(np.uint8)
    
    # Morphological closing
    closed = merged_bin
    for ksize in morph_kernels:
        kernel = np.ones(ksize, np.uint8)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    
    # Clean small islands
    final_cleaned = remove_small_islands(closed, max_size=island_size)
    
    # Connected components
    inverted = cv2.bitwise_not(final_cleaned)
    _, labels = cv2.connectedComponents(inverted, connectivity=8)
    
    # Region filtering
    labels_step = labels.copy()
    for region in regionprops(labels_step):
        if region.area < min_area or region.extent < extent_thresh:
            labels_step[labels_step == region.label] = 0
    filtered_labels = label(labels_step > 0)
    
    # Remove the extra padding from the image
    # Get original image dimensions
    h, w = img.shape

    # Crop filtered_labels to original dimensions
    filtered_labels = filtered_labels[:h, :w]


    # Evaluation logic
    if gt_mat_path is not None:
        # Supervised evaluation
        gt = scipy.io.loadmat(gt_mat_path)['indian_pines_gt']
        metrics = evaluate_segmentation(filtered_labels, gt, iou_threshold)
    else:
        # Unsupervised evaluation
        y_coords, x_coords = np.indices((h, w))
        features = np.stack([
            x_coords.ravel()/w,          # Normalized X
            y_coords.ravel()/h,          # Normalized Y
            img.ravel()/255.0            # Normalized intensity
        ], axis=1)
        
        flat_labels = filtered_labels.ravel()
        unique_labels = np.unique(flat_labels)
        
        if len(unique_labels) > 1:
            metrics = {
                'silhouette': silhouette_score(features, flat_labels),
                'calinski': calinski_harabasz_score(features, flat_labels),
                'davies': davies_bouldin_score(features, flat_labels)
            }
        else:
            metrics = {
                'silhouette': -1,
                'calinski': 0,
                'davies': float('inf')
            }

    
    if visualize:
        preview_images(
            visualize_labels(filtered_labels), 
            titles=[f"Tile: {tile_size}, Overlap: {overlap}"],
            figsize=(5,5)
        )
    
    return metrics, filtered_labels


def calculate_parameters(image_path, base_params):
    """Dynamically generate parameters based on image dimensions"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    max_dim = max(h, w)
    
    # Generate tile sizes from 32px up to full image size
    tile_sizes = []
    current_size = base_params['min_tile']
    while current_size <= max_dim:
        tile_sizes.append((current_size, current_size))
        if current_size == max_dim: break
        current_size = min(current_size + base_params['tile_step'], max_dim)
    
    # Generate overlaps as percentages of tile size
    param_combos = []
    for ts in tile_sizes:
        for op in base_params['overlap_percentages']:
            overlap = int(ts[0] * op)
            if overlap < ts[0]:  # Valid overlap
                param_combos.append({'tile_size': ts, 'overlap': overlap})
    
    # Add full-image special case
    param_combos.append({'tile_size': (h,w), 'overlap': 0})
    
    return param_combos


def automated_grid_search(
    image_path,
    gt_mat_path=None,  # Now optional (None for unsupervised)
    base_params=None,
    unsupervised_metric='silhouette'  # New parameter to choose metric
):
    """Smart parameter exploration with adaptive ranges
    
    Now handles both supervised and unsupervised evaluation:
    - With GT: Sorts by F1-score (supervised)
    - Without GT: Sorts by chosen clustering metric (silhouette/calinski/davies)
    """
    # Determine evaluation mode based on GT presence
    supervised_mode = gt_mat_path is not None
    
    # Get parameter combinations (same as before)
    all_params = calculate_parameters(image_path, base_params)
    results = []
    
    # Grid search loop (same as before)
    for params in all_params:
        print(f"\nTesting Tile: {params['tile_size']}, Overlap: {params['overlap']}")
        metrics, _ = segmentation_pipeline(
            image_path,
            gt_mat_path,  # Passes None automatically if not provided
            **params,
            visualize=base_params.get('visualize', False)
        )
        results.append({
            'tile_size': params['tile_size'],
            'overlap': params['overlap'],
            **metrics
        })
    
    # Determine sorting strategy
    if supervised_mode:
        # Supervised: Sort by F1-score (descending)
        sort_key = 'f1_score'
        reverse = True
    else:
        # Unsupervised: Use selected metric
        sort_key = unsupervised_metric
        
        # Reverse sorting depends on metric type:
        # - Higher is better for silhouette & calinski
        # - Lower is better for davies
        reverse = unsupervised_metric != 'davies'
    
    return sorted(results, key=lambda x: x.get(sort_key, 0), reverse=reverse)


# # Execute search
def run_segmentation(
        input_image='output_pca.tiff',
        ground_truth=None,
        search_config=None,
        unsupervised_metric='silhouette',
        print_results=True
):
    """
    Run segmentation pipeline with configurable parameters.

    Example usage:
    
    1. Supervised evaluation with ground truth:
    ```
       results = run_segmentation(
           input_image='output_pca.tiff',
           ground_truth='Indian_pines_gt.mat',
           search_config={
               'min_tile': 8,
               'tile_step': 8,
               'overlap_percentages': [0, 0.1, 0.25, 0.33, 0.5],
               'visualize': False
           },
           print_results=True
       )
       ```

    2. Unsupervised evaluation without ground truth:
         ```
       results = run_segmentation(
           input_image='output_pca.tiff',
           ground_truth=None,
           search_config={
               'min_tile': 8,
               'tile_step': 8,
               'overlap_percentages': [0, 0.1, 0.25, 0.33, 0.5],
               'visualize': False
           },
           unsupervised_metric='silhouette',  # (silhouette/calinski/davies)
           print_results=True
       )
         ```

    Args:
        input_image (str): Path to input image file.
        ground_truth (str): Path to ground truth .mat file (optional).
        search_config (dict): Dictionary with grid search parameters. Example:
            search_config = {
                'min_tile': 8,        # Start from 32x32 tiles
                'tile_step': 8,       # Increase tile size by 32px each step
                'overlap_percentages': [0, 0.1, 0.25, 0.33, 0.5],  # 0-50% of tile size
                'visualize': False    # Show top 3 results
            }
        unsupervised_metric (str): Metric for evaluation without GT.
        print_results (bool): Whether to print formatted results or just return them.

    Returns:
        list: List of result dictionaries sorted by best performance.
    """
    
    # Handle mutable default safely
    if search_config is None:
        search_config = {
            'min_tile': 8,
            'tile_step': 8,
            'overlap_percentages': [0, 0.1, 0.25, 0.33, 0.5],
            'visualize': False
        }

    # Validate critical parameters
    if not isinstance(search_config, dict):
        raise ValueError("search_config must be a dictionary")
        
    # Run grid search
    results = automated_grid_search(
        image_path=input_image,
        gt_mat_path=ground_truth,
        base_params=search_config,
        unsupervised_metric=unsupervised_metric
    )

    # Early return if not printing
    if not print_results:
        return results

    # Handle empty results
    if not results:
        print("No results to display")
        return results

    # Determine evaluation mode safely
    is_supervised = 'f1_score' in results[0] if results else False

    # Print results with consistent formatting
    print("\nOptimization Results:")
    
    if is_supervised:
        print(f"{'Tile Size':<12} | {'Overlap':<8} | {'F1-Score':<8} | {'Precision':<10} | {'Recall':<8}")
        for res in results[:10]:
            print(f"{str(res['tile_size']):<12} | "
                  f"{res['overlap']:<8} | "
                  f"{res.get('f1_score', 0):.3f}      | "
                  f"{res.get('precision', 0):.3f}       | "
                  f"{res.get('recall', 0):.3f}")
    else:
        print(f"{'Tile Size':<12} | {'Overlap':<8} | {'Silhouette':<10} | {'Calinski':<10} | {'Davies':<10}")
        for res in results[:10]:
            print(f"{str(res['tile_size']):<12} | "
                  f"{res['overlap']:<8} | "
                  f"{res.get('silhouette', -1):>10.3f} | "
                  f"{res.get('calinski', 0):>10.3f} | "
                  f"{res.get('davies', float('inf')):>10.3f}")

    return results
