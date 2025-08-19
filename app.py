import streamlit as st
import numpy as np
import cv2
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Function to create a simple sample image (a gradient with a circle)
def create_sample_image():
    """Generates a sample RGB image with a gradient background and a white circle.

    Returns a PIL RGB Image. The processing pipeline in this demo still runs on a
    grayscale copy derived from the color image.
    """
    width, height = 300, 300
    gray = np.zeros((height, width), dtype=np.uint8)
    # Create a simple horizontal gradient
    for i in range(width):
        gray[:, i] = np.linspace(0, 255, height, dtype=np.uint8)

    # Draw a white circle in the center
    center_x, center_y = width // 2, height // 2
    radius = 60
    for x in range(width):
        for y in range(height):
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                gray[y, x] = 255

    # Make an RGB image by stacking the gray channel
    rgb = np.stack([gray, gray, gray], axis=2)
    return Image.fromarray(rgb)

# --- Point Operations (Spatial Domain) ---

def apply_image_negative(image):
    """Applies the image negative transformation."""
    return cv2.bitwise_not(image)

def apply_contrast_stretching(image):
    """Applies a simple contrast stretching operation."""
    min_val, max_val, _, _ = cv2.minMaxLoc(image)
    if (max_val - min_val) > 0:
        processed_image = 255 * (image - min_val) / (max_val - min_val)
        return processed_image.astype(np.uint8)
    return image

def apply_thresholding(image, threshold_value):
    """Applies simple binary thresholding."""
    _, processed_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return processed_image

def apply_window_slicing(image, min_val, max_val):
    """Highlights pixels within a specific intensity range."""
    processed_image = np.zeros_like(image)
    processed_image[np.where((image >= min_val) & (image <= max_val))] = 255
    return processed_image

def apply_histogram_equalization(image):
    """Applies histogram equalization."""
    return cv2.equalizeHist(image)

def apply_log_transformation(image, c):
    """Applies a log transformation to compress dynamic range."""
    processed_image = c * np.log1p(image)
    processed_image = np.interp(processed_image, (processed_image.min(), processed_image.max()), (0, 255)).astype(np.uint8)
    return processed_image

def apply_power_law_transformation(image, gamma):
    """Applies a power-law transformation (gamma correction)."""
    processed_image = np.power(image / 255.0, gamma)
    processed_image = (processed_image * 255).astype(np.uint8)
    return processed_image

def apply_bit_plane_slicing(image, bit_plane):
    """Extracts a specific bit plane from the image."""
    processed_image = np.bitwise_and(image, 2**bit_plane) * 255
    return processed_image.astype(np.uint8)

def apply_image_subtraction(image):
    """Subtracts a second image (a simple dark square) for demonstration."""
    sub_image = np.zeros_like(image)
    sub_image[100:200, 100:200] = 50
    processed_image = cv2.subtract(image, sub_image)
    return processed_image

def apply_image_multiplication(image):
    """Multiplies with a second image (a simple bright square) for masking."""
    mult_image = np.zeros_like(image)
    mult_image[100:200, 100:200] = 255
    processed_image = cv2.multiply(image, mult_image)
    return processed_image

# --- Spatial Operations ---

def apply_spatial_averaging(image):
    """Applies a simple blurring filter for spatial averaging."""
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

def apply_median_blur(image):
    """Applies a median blur filter, effective for salt-and-pepper noise."""
    return cv2.medianBlur(image, 5)

def apply_unsharp_masking(image, k):
    """
    Applies unsharp masking by subtracting a blurred version of the image.
    k controls the amount of sharpening.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 5)
    processed_image = cv2.addWeighted(image, 1 + k, blurred, -k, 0)
    return processed_image

def apply_min_filter(image):
    """Applies a minimum filter to find the darkest pixel in a neighborhood."""
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_max_filter(image):
    """Applies a maximum filter to find the brightest pixel in a neighborhood."""
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
    
def apply_geometric_mean_filter(image):
    """Applies the geometric mean filter."""
    def geometric_mean(array):
        # Avoid issues with log(0)
        array[array == 0] = 1 
        return np.prod(array) ** (1.0 / len(array))

    rows, cols = image.shape
    new_image = np.zeros_like(image)
    
    # Pad the image to handle borders
    padded_image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REFLECT)
    
    for i in range(rows):
        for j in range(cols):
            window = padded_image[i:i+5, j:j+5].flatten()
            new_image[i, j] = geometric_mean(window)
            
    return new_image.astype(np.uint8)

# --- Transform Operations (Frequency Domain) ---

def apply_high_pass_filter(image):
    """
    Applies a simple high-pass filter in the frequency domain.
    This enhances edges and fine details.
    """
    f = fft2(image.astype(float))
    fshift = fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a circular high-pass filter mask
    mask = np.ones((rows, cols), np.uint8)
    r = 30 # Radius of the filter
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2
    mask[mask_area] = 0
    
    # Apply the mask and transform back
    fshift = fshift * mask
    f_ishift = ifftshift(fshift)
    processed_image = ifft2(f_ishift)
    processed_image = np.abs(processed_image)
    
    # Normalize the output to the full 8-bit range
    processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)
    return processed_image.astype(np.uint8)

def apply_homomorphic_filter(image):
    """Applies a simplified homomorphic filter for illumination correction."""
    # 1. Log transformation
    log_img = np.log1p(image.astype(float))
    
    # 2. Fourier Transform
    f = fft2(log_img)
    fshift = fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # 3. Create a high-pass filter (Gaussian)
    D = np.sqrt((np.arange(rows) - crow)[:, np.newaxis]**2 + (np.arange(cols) - ccol)**2)
    cutoff = 50
    gamma_l = 0.5  # Lower gamma for low frequencies
    gamma_h = 2.0  # Higher gamma for high frequencies
    H = (gamma_h - gamma_l) * (1 - np.exp(-(D**2) / (2 * cutoff**2))) + gamma_l
    
    # 4. Filter the image
    filtered_fshift = fshift * H
    
    # 5. Inverse Fourier Transform
    f_ishift = ifftshift(filtered_fshift)
    processed_image = np.abs(ifft2(f_ishift))
    
    # 6. Exponentiation
    processed_image = np.expm1(processed_image)
    
    # Normalize to 0-255 range
    processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)
    return processed_image.astype(np.uint8)

def apply_root_filter(image, alpha):
    """Applies a Root filter in the frequency domain."""
    f = fft2(image.astype(float))
    magnitude = np.abs(f)
    phase = np.angle(f)
    
    # Apply power-law to magnitude
    new_magnitude = magnitude ** alpha
    
    # Combine new magnitude and original phase
    filtered_f = new_magnitude * np.exp(1j * phase)
    
    # Inverse Fourier Transform
    processed_image = np.abs(ifft2(filtered_f))
    
    # Normalize to 0-255 range
    processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)
    return processed_image.astype(np.uint8)
    
def apply_inverse_filter(image):
    """
    (Placeholder) Inverse filtering requires a known degradation function,
    which is not available in this demo. This function provides a conceptual
    result, but a full implementation would need more context.
    """
    st.warning("Inverse Filtering is a complex restoration task. A full implementation requires a known degradation function (e.g., blur). This operation will return the original image as a placeholder.")
    return image

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Image Processing App")
st.title("Interactive Image Processing App")
st.markdown("Use the controls below to apply different image processing operations to a sample image.")

# Define the operations in a dictionary for easy mapping
operations = {
    "Point Operations (Spatial Domain)": {
        "Image Negative": apply_image_negative,
        "Contrast Stretching": apply_contrast_stretching,
        "Thresholding": apply_thresholding,
        "Window Slicing": apply_window_slicing,
        "Histogram Equalization": apply_histogram_equalization,
        "Log Transformations": apply_log_transformation,
        "Power-Law Transformations": apply_power_law_transformation,
        "Bit-Plane Slicing": apply_bit_plane_slicing,
        "Image Subtraction": apply_image_subtraction,
        "Image Multiplication": apply_image_multiplication,
    },
    "Spatial Operations": {
        "Spatial Averaging": apply_spatial_averaging,
        "Median Filtering": apply_median_blur,
        "Unsharp Masking": apply_unsharp_masking,
        "Min Filter": apply_min_filter,
        "Max Filter": apply_max_filter,
        "Geometric Mean Filter": apply_geometric_mean_filter,
    },
    "Transform Operations (Frequency Domain)": {
        "High-pass Filtering": apply_high_pass_filter,
        "Homomorphic Filtering": apply_homomorphic_filter,
        "Root Filtering": apply_root_filter,
        "Inverse Filtering": apply_inverse_filter,
    },
}

# Initialize session state for the image if it doesn't exist
if 'current_image_color' not in st.session_state or 'current_image_np' not in st.session_state:
    sample_color = np.array(create_sample_image())  # RGB
    sample_gray = cv2.cvtColor(sample_color, cv2.COLOR_RGB2GRAY)
    st.session_state.current_image_color = sample_color
    st.session_state.current_image_np = sample_gray

# Create a container for the controls in a sidebar
with st.sidebar:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Decode as color (BGR) then convert to RGB for display
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            st.session_state.current_image_color = img_rgb
            st.session_state.current_image_np = img_gray
    
    st.header("Operation Controls")
    
    # Dropdown for the operation type
    operation_type = st.selectbox(
        "Select Operation Type",
        list(operations.keys())
    )
    
    # Dropdown for the specific operation, dynamically populated
    selected_operation = st.selectbox(
        "Select Specific Operation",
        list(operations[operation_type].keys())
    )

    # --- Conditional Sliders for Parameters ---
    if selected_operation in ["Thresholding"]:
        threshold_value = st.slider("Threshold Value", 0, 255, 127)
    if selected_operation in ["Window Slicing"]:
        min_val = st.slider("Minimum Intensity", 0, 255, 50)
        max_val = st.slider("Maximum Intensity", 0, 255, 150)
    if selected_operation in ["Log Transformations"]:
        c = st.slider("Scaling Constant (c)", 1.0, 10.0, 1.0)
    if selected_operation in ["Power-Law Transformations"]:
        gamma = st.slider("Gamma Value", 0.1, 5.0, 1.0)
    if selected_operation in ["Bit-Plane Slicing"]:
        bit_plane = st.slider("Bit Plane (0-7)", 0, 7, 0)
    if selected_operation in ["Unsharp Masking"]:
        k = st.slider("Sharpening Amount (k)", 0.0, 5.0, 1.0)
    if selected_operation in ["Root Filtering"]:
        alpha = st.slider("Root Alpha (α)", 0.01, 2.0, 1.0)
    
    # Try it button
    try_button = st.button("Try It")

# Create columns for displaying the images
col1, col2 = st.columns(2)

# Display the original image (color) from session state
with col1:
    st.header("Original Image (Color)")
    st.image(st.session_state.current_image_color, use_container_width=True)

# Process and display the image when the button is clicked
if try_button:
    # Get the function to apply and its arguments
    processing_function = operations[operation_type][selected_operation]
    
    # Apply the selected function with its corresponding parameters
    try:
        if selected_operation == "Thresholding":
            processed_image_np = processing_function(st.session_state.current_image_np, threshold_value)
        elif selected_operation == "Window Slicing":
            processed_image_np = processing_function(st.session_state.current_image_np, min_val, max_val)
        elif selected_operation == "Log Transformations":
            processed_image_np = processing_function(st.session_state.current_image_np, c)
        elif selected_operation == "Power-Law Transformations":
            processed_image_np = processing_function(st.session_state.current_image_np, gamma)
        elif selected_operation == "Bit-Plane Slicing":
            processed_image_np = processing_function(st.session_state.current_image_np, bit_plane)
        elif selected_operation == "Unsharp Masking":
            processed_image_np = processing_function(st.session_state.current_image_np, k)
        elif selected_operation == "Root Filtering":
            processed_image_np = processing_function(st.session_state.current_image_np, alpha)
        else:
            processed_image_np = processing_function(st.session_state.current_image_np)

        processed_image_pil = Image.fromarray(processed_image_np)
        
        with col2:
            st.header(f"Processed Image ({selected_operation})")
            st.image(processed_image_pil, use_container_width=True)
            st.success("Operation successful!")
    except Exception as e:
        with col2:
            st.error(f"An error occurred: {e}")
