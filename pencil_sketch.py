import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

# Function to perform color quantization using K-means clustering
def quantize_colors(image, k=5):
    reshaped_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_image)
    quantized_colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    quantized_image = quantized_colors[labels].reshape(image.shape)
    return quantized_image
# Function to perform color transfer from the style image to the input image
def transfer_colors(input_image, style_image):
    input_lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    style_lab = cv2.cvtColor(style_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Compute mean and standard deviation of color channels
    input_mean = np.mean(input_lab, axis=(0, 1))
    style_mean = np.mean(style_lab, axis=(0, 1))
    input_std = np.std(input_lab, axis=(0, 1))
    style_std = np.std(style_lab, axis=(0, 1))
    
    # Perform color transfer
    transferred_lab = input_lab - input_mean
    transferred_lab *= (style_std / input_std)
    transferred_lab += style_mean
    
    # Clip the pixel values to the valid range [0, 255]
    transferred_lab = np.clip(transferred_lab, 0, 255).astype(np.uint8)
    
    # Convert back to BGR color space
    transferred_image = cv2.cvtColor(transferred_lab, cv2.COLOR_LAB2BGR)
    
    return transferred_image

# Function to convert image to pencil sketch
def pencil_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    inverted_blurred = 255 - blurred
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    return pencil_sketch

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the input image file
input_image_path = os.path.join(current_dir, "tiger.jpg")

# Construct the absolute path to the style image file
style_image_path = os.path.join(current_dir, "starry_night.jpg")

# Check if the input image file exists
if not os.path.isfile(input_image_path):
    print(f"File '{input_image_path}' does not exist.")
    exit()

# Check if the style image file exists
if not os.path.isfile(style_image_path):
    print(f"File '{style_image_path}' does not exist.")
    exit()

# Load the input image
input_image = cv2.imread(input_image_path)

# Load the style image
style_image = cv2.imread(style_image_path)

# Perform color quantization on the style image
quantized_style_image = quantize_colors(style_image)

# Transfer colors from the quantized style image to the input image
transferred_image = transfer_colors(input_image, quantized_style_image)

# Convert input image to pencil sketch
sketch_image = pencil_sketch(input_image)

# Convert sketch image to three-dimensional array with single channel
sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_GRAY2BGR)

# Resize the input, transferred, and sketch images for better visualization
resize_factor = 1.3
input_image_resized = cv2.resize(input_image, None, fx=resize_factor, fy=resize_factor)
transferred_image_resized = cv2.resize(transferred_image, None, fx=resize_factor, fy=resize_factor)
sketch_image_resized = cv2.resize(sketch_image, None, fx=resize_factor, fy=resize_factor)

# Concatenate the original input, transferred, and sketch images horizontally
concatenated_image = np.concatenate((input_image_resized, transferred_image_resized, sketch_image_resized), axis=1)

# Display the concatenated image
cv2.imshow("Input Image vs Transferred Image vs Pencil Sketch", concatenated_image)

# Wait for any key to be pressed
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()