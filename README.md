# Patch-Based Image Segmentation Project

## Project Overview
This project implements a web-based image segmentation system that compares three different segmentation methods: K-means clustering, Multi-Otsu thresholding, and traditional thresholding. The system uses a patch-based approach to process images and provides a comprehensive analysis of the results.

## Key Features Implemented

1. **Web Interface**
   - Flask-based web application
   - User-friendly interface for image upload
   - Real-time processing and results display

2. **Image Processing**
   - Automatic optimal patch size determination
   - Grayscale conversion for processing
   - Patch extraction and processing

3. **Segmentation Methods**
   - K-means clustering implementation
   - Multi-Otsu thresholding with error handling
   - Traditional thresholding based on mean intensity

4. **Performance Evaluation**
   - Mean Square Error (MSE) calculation
   - Peak Signal-to-Noise Ratio (PSNR) computation
   - Comparative analysis of segmentation methods

5. **Results Visualization**
   - Original image display
   - Segmented results for all three methods
   - Performance metrics table
   - Best method identification

6. **Technical Implementation**
   - Python-based backend
   - HTML/CSS frontend
   - Image processing using OpenCV and scikit-image
   - Statistical analysis using scikit-learn

7. **Error Handling**
   - File type validation
   - Image processing error handling
   - Multi-Otsu thresholding fallback mechanism

8. **Project Structure**
   - Modular code organization
   - Separate directories for uploads and results
   - Template-based web interface

## Research Paper Components
The accompanying research paper covers:
1. Mathematical foundations of each segmentation method
2. Detailed explanation of patch-based processing
3. Performance metrics and evaluation methodology
4. Comparative analysis of results
5. Future work and potential improvements

## Requirements
- Python 3.x
- Flask
- OpenCV
- scikit-image
- scikit-learn
- NumPy
- Pillow
- Matplotlib

## Usage
1. Install required dependencies
2. Run the Flask application
3. Access the web interface
4. Upload an image for processing
5. View and analyze the results 