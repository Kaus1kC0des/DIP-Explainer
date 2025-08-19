# Image Processing App — Detailed README

This README explains the code in `main.py` in detail, covering the implementation, the mathematical background for each image processing operation, usage instructions, and notes on limitations and possible improvements.

## Quick overview

- Project entry: `main.py` — a Streamlit app demonstrating a variety of image processing operations implemented in Python using OpenCV, NumPy, SciPy (FFT), and PIL.
- Purpose: Interactive exploration of point operations, spatial filters, and frequency-domain transforms with adjustable parameters.

## Table of contents

- Checklist (what this README covers)
- How to run
- Dependencies
- File structure
- Code walkthrough (function-by-function)
  - Image generation
  - Point operations (spatial domain)
  - Spatial operations (local filters)
  - Transform operations (frequency domain)
  - Streamlit UI and control flow
- Math appendix (formulas, derivations, discretizations)
- Limitations and suggested improvements

## Checklist — user's request coverage

- Create README.md explaining all code: Done.
- Explain math behind each transform: Done (see Math appendix and inline explanations).
- Show how to run and dependencies: Done.

## How to run

1. From the project root (where `main.py` is located) install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. The app opens in your browser. Use the sidebar to upload an image (or use the sample image), choose operation categories, tweak parameters, and click "Try It" to see results.

## Dependencies

- Python 3.8+ (recommended)
- streamlit
- numpy
- opencv-python
- pillow (PIL)
- scipy (for FFT functions `fft2`, `ifft2`, and helpers)

Install with the pip command shown above.

## File structure

- `main.py` — single-file Streamlit app containing all UI code and processing functions.
- `README.md` — this file (explainers and math).

## Code walkthrough — function-by-function

All references are to `main.py`.

### Sample image generator

Function: `create_sample_image()`

- Purpose: build a small grayscale 300×300 sample image with a horizontal gradient and a centered white circle. Useful when no image is uploaded.
- Implementation details: uses NumPy to create an array and fills columns with a linear gradient. The circle is drawn by checking the implicit circle equation:

  (x - x_c)^2 + (y - y_c)^2 < r^2

  Pixels satisfying this inequality are set to 255 (white).

### Point Operations (Spatial Domain)

These operate on pixel intensities independently (no neighborhood).

1. `apply_image_negative(image)`
   - Implementation: `cv2.bitwise_not(image)`.
   - Math: s = 255 - r for 8-bit images, where r is the input pixel intensity and s is the output.

2. `apply_contrast_stretching(image)`
   - Implementation: find min and max pixel values (r_min, r_max) and linearly map to full range:

     s = 255 * (r - r_min) / (r_max - r_min)

   - This is a linear contrast stretch; avoids division-by-zero if image is constant.

3. `apply_thresholding(image, threshold_value)`
   - Implementation: `cv2.threshold(..., cv2.THRESH_BINARY)`.
   - Math: s = 255 if r >= T else 0.

4. `apply_window_slicing(image, min_val, max_val)`
   - Implementation: pixels inside [min_val, max_val] are set to 255, others to 0.
   - Use-case: highlight intensity ranges.

5. `apply_histogram_equalization(image)`
   - Implementation: `cv2.equalizeHist(image)`.
   - Math: map intensity r to s using the image's cumulative distribution function (CDF):

     s = floor( (L - 1) * CDF(r) )

     where L = number of intensity levels (256 for 8-bit). This remaps intensities to flatten the histogram and improve global contrast.

6. `apply_log_transformation(image, c)`
   - Implementation: processed_image = c * log(1 + r), then rescaled to 0–255.
   - Math: s = c * log(1 + r). This compresses high-intensity values and expands low-intensity values (useful for images with large dynamic range).

7. `apply_power_law_transformation(image, gamma)`
   - Implementation: s = (r/255)^{gamma} * 255.
   - Math: power-law (gamma) correction. If gamma < 1, the image brightens; gamma > 1 darkens.

8. `apply_bit_plane_slicing(image, bit_plane)`
   - Implementation: extract bit by `np.bitwise_and(image, 2**bit_plane) * 255`.
   - Math: For an 8-bit pixel r = sum_{k=0..7} b_k 2^k. The k-th bit plane is b_k.

9. `apply_image_subtraction(image)`
   - Implementation: subtract a constant dark square (50) at a fixed region using `cv2.subtract`.
   - Math: s(x,y) = max(0, r(x,y) - d(x,y)).

10. `apply_image_multiplication(image)`
    - Implementation: multiply pixel-wise by a mask square that is 255 in a region and 0 elsewhere using `cv2.multiply`.
    - Math: s(x,y) = r(x,y) * m(x,y)/255 (OpenCV handles scaling), serves as masking.

### Spatial Operations (Local neighborhood filters)

These operations use a neighborhood (kernel/window) around each pixel.

1. `apply_spatial_averaging(image)`
   - Implementation: convolution with a 5×5 kernel of ones normalized by 25 using `cv2.filter2D`.
   - Math: s(x,y) = (1/25) Σ_{i=-2..2} Σ_{j=-2..2} r(x+i, y+j). A low-pass filter (blurs image).

2. `apply_median_blur(image)`
   - Implementation: `cv2.medianBlur(image, 5)`.
   - Math: s(x,y) = median of pixels in a 5×5 neighborhood. Robust to salt-and-pepper noise.

3. `apply_unsharp_masking(image, k)`
   - Implementation: blurred = GaussianBlur(image, sigma=5), then processed = addWeighted(image, 1+k, blurred, -k, 0).
   - Math: s = (1 + k) I - k B = I + k (I - B), where I is the original image and B is the blurred image. This adds a scaled high-frequency (detail) term, enhancing edges.

4. `apply_min_filter(image)`
   - Implementation: morphological erosion via kernel (5×5 ones) — finds local minimum in window.
   - Math: s(x,y) = min_{(i,j) in window} r(x+i,y+j).

5. `apply_max_filter(image)`
   - Implementation: morphological dilation via kernel (5×5 ones) — finds local maximum in window.
   - Math: s(x,y) = max_{(i,j) in window} r(x+i,y+j).

6. `apply_geometric_mean_filter(image)`
   - Implementation: iterate each pixel, extract 5×5 window, compute product and take the N-th root (where N=25). Zero values are replaced with 1 to avoid log(0) or product=0.
   - Math: s = (Π_{n=1..N} r_n)^{1/N}. More suitable for multiplicative noise reductions. Implemented naively (loops); relatively slow but straightforward.

### Transform Operations (Frequency Domain)

These operations use the Fourier transform. Implementation relies on SciPy's `fft2`/`ifft2` and `fftshift`/`ifftshift`.

Definitions and notation used:

- Let f(x,y) be the spatial-domain image (real-valued intensity). The 2D discrete Fourier transform (DFT) is:

  F(u, v) = Σ_{x=0..M-1} Σ_{y=0..N-1} f(x,y) e^{-j2π(ux/M + vy/N)}

  and the inverse DFT is:

  f(x, y) = (1/MN) Σ_{u=0..M-1} Σ_{v=0..N-1} F(u,v) e^{j2π(ux/M + vy/N)}

- In practice the code uses fast FFT implementations and shifts the zero-frequency component to the center of the spectrum using `fftshift`.

1. `apply_high_pass_filter(image)`
   - Implementation summary:
     - Compute f = fft2(image)
     - Shift zero-frequency to center: fshift
     - Create a circular mask that zeros out the low frequencies inside radius r (r = 30 by default). The rest is kept.
     - Multiply the shifted spectrum by mask, inverse-shift and inverse-FFT, take magnitude and normalize to 0–255.
   - Math: High-pass filtering in frequency domain corresponds to multiplying the spectrum by a filter H(u,v) that is close to 0 at low frequencies and 1 at high frequencies. The mask used is an ideal high-pass filter:

     H(u,v) = 0 for D(u,v) <= r
     H(u,v) = 1 for D(u,v) > r

     where D(u,v) is the distance from the center frequency.

   - Notes: Ideal filters cause ringing in spatial domain (Gibbs phenomenon) but are simple to demonstrate.

2. `apply_homomorphic_filter(image)`
   - Purpose: correct non-uniform illumination by treating illumination and reflectance multiplicatively: f(x,y) = i(x,y) * r(x,y). We apply a log transform to convert multiplicative components into additive ones:

     ln f = ln i + ln r

   - Implementation summary:
     - 1) log transform: log_img = ln(1 + image)
     - 2) FFT and shift
     - 3) Create a Gaussian-shaped high-pass filter H(D):

       H(D) = (γ_H - γ_L) (1 - exp(-D^2 / (2 D0^2))) + γ_L

       where D is distance from center, D0 is cutoff (called `cutoff` in code, set to 50), γ_L < 1 (attenuate low frequencies/illumination), γ_H > 1 (boost high frequencies/reflectance details).

     - 4) Multiply spectrum by H, inverse FFT and shift, exponentiate (exp(processed)-1) to return to intensity domain, then normalize.

   - Math details:
     - The filter is designed so that low-frequency components (associated with illumination) are reduced (multiplied by γ_L) and high-frequency components (details/reflectance) are amplified (multiplied by γ_H). The Gaussian-shaped transition provides a smooth frequency response.

3. `apply_root_filter(image, alpha)`
   - Purpose: apply a power-law (root/power) operation in the frequency domain on the magnitude spectrum while preserving phase.
   - Implementation summary:
     - f = fft2(image); magnitude = |f|; phase = angle(f)
     - new_magnitude = magnitude ** alpha
     - reconstructed spectrum = new_magnitude * exp(j * phase)
     - inverse FFT, magnitude, normalize to 0–255

   - Math: This manipulates the spectral energy distribution. When alpha < 1, it compresses large magnitudes; alpha > 1 accentuates strong spectral components. Keeping the original phase preserves spatial features' locations while changing contrast/texture.

4. `apply_inverse_filter(image)`
   - Implementation: placeholder that returns original image and a Streamlit warning.
   - Explanation: True inverse filtering requires a known degradation function H(u,v) (e.g., blur kernel in frequency domain). Inverse filtering formula given known H is:

     F_hat(u,v) = G(u,v) / H(u,v)

     where G is degraded/image spectrum and H is degradation. Practical implementations must protect against zeros in H and noise amplification (Wiener filtering is commonly used instead).

## Streamlit UI and control flow

- App layout is wide with a sidebar for controls.
- `operations` dict maps three top-level categories to named functions; the UI dynamically populates specific operations based on category.
- Session state key: `st.session_state.current_image_np` stores the current grayscale image as a NumPy 2D array. On first load, it is initialized with the sample image returned from `create_sample_image().convert('L')`.
- Upload: `st.file_uploader` reads an uploaded image, decodes it into a grayscale OpenCV image via `cv2.imdecode(..., 0)` and stores it in session state.
- Parameter controls: sliders are conditionally displayed for operations that take parameters (threshold, window slicing bounds, log `c`, gamma, bit plane, unsharp `k`, root alpha, etc.).
- Execution: when the user clicks "Try It", the app looks up the function in `operations` and calls it with appropriate parameters. The processed NumPy array is converted to a PIL image via `Image.fromarray` and displayed.
- Errors are caught and shown in the UI.

## Math appendix — formulas and short derivations

This appendix collects the core mathematical formulas referenced in the code. Where appropriate the 8-bit pixel range [0,255] is used for discretization.

1) Negative transformation

- s = L - 1 - r

  For L = 256 (8-bit), s = 255 - r.

2) Contrast stretching (linear normalization)

- Given r_min, r_max in the image, map r → s by

  s = (r - r_min) * (L - 1) / (r_max - r_min)

3) Thresholding

- s = 255 if r >= T; else s = 0.

4) Histogram equalization

- Let p_r(r_k) be the (normalized) histogram probability for intensity r_k, k = 0..L-1.
- The cumulative distribution function (CDF) is

  CDF(r_k) = Σ_{i=0..k} p_r(r_i)

- The mapping is

  s_k = floor( (L - 1) * CDF(r_k) )

5) Log transformation

- s = c * ln(1 + r), often followed by scaling to 0..255. Using ln(1+r) avoids ln(0).

6) Gamma (power-law) transformation

- s = c * (r / (L - 1))^{γ} * (L - 1)  (with c typically = 1 in the code)

- In the code we use r/255 then power and multiply by 255.

7) Bit-plane slicing

- For r = Σ b_k 2^k, the k-th bit plane is b_k.

8) Local filters (linear convolution)

- Convolution: s(x,y) = Σ_{i} Σ_{j} h(i,j) r(x-i, y-j).
- Spatial averaging uses uniform kernel h = 1/N.

9) Unsharp masking

- s = I + k (I - B) where B = GaussianBlur(I). The (I - B) term approximates high-frequency content.

10) Geometric mean filter

- s = (Π_{n=1..N} r_n)^{1/N}.

- Numerically safer via logs: s = exp( (1/N) Σ log(r_n) ). Note: code sets zeros to 1 to avoid log(0).

11) Fourier transform (discrete)

- 2D DFT:

  F(u,v) = Σ_{x=0..M-1} Σ_{y=0..N-1} f(x,y) e^{-j2π(ux/M + vy/N)}

  f(x,y) = (1/MN) Σ_{u=0..M-1} Σ_{v=0..N-1} F(u,v) e^{j2π(ux/M + vy/N)}

12) Ideal high-pass filter

- H(u,v) = 0 if D(u,v) <= r else 1, where D is distance from center.

13) Homomorphic filter (used in code)

- Working in the log domain: S = ln f = ln i + ln r
- Filter design: H(D) = (γ_H - γ_L) (1 - e^{-D^2/(2 D0^2)}) + γ_L
- Apply to spectrum, inverse transform, then exponentiate: f' = exp( filtered(S) ) - 1

14) Root/power filtering in frequency domain

- If F = |F| e^{jφ}, then the filtered spectrum is |F|^{α} e^{jφ}. Inverse transform gives a modified spatial image where spectral amplitudes are altered.

## Implementation notes, pitfalls and numerical details

- Data types: FFT operations are done on float arrays. After inverse transforms we take magnitudes and normalize to 0–255 using `cv2.normalize(..., cv2.NORM_MINMAX)` to map to displayable 8-bit images.
- Phase preservation: when manipulating the spectrum, preserving phase is critical to maintain spatial structure; changing magnitude alone modifies contrast/texture.
- Edge effects: frequency-domain ideal filters (sharp masks) can induce ringing artifacts. Gaussian or Butterworth filters are smoother alternatives.
- Geometric mean filter implementation is O(MN * N_window) due to inner loops. Consider using vectorized or log-domain implementations for speed.
- Homomorphic filtering parameters (γ_L, γ_H, cutoff) strongly influence output; they may need tuning per image.

## Limitations and suggested improvements

1. Performance
   - The `apply_geometric_mean_filter` is implemented with Python loops and will be slow for large images. Replace with a vectorized, block-based method or use log-domain convolution.

2. Robustness
   - In `apply_inverse_filter`, a real implementation needs an estimated or known degradation function H and regularization (e.g., Wiener filter) to avoid noise amplification.

3. UI/UX
   - Allow choosing filter radii, kernel sizes, Gaussian sigma, and homomorphic parameters interactively.
   - Add side-by-side histograms (original vs processed) and spectrum visualizations (log-magnitude) for teaching purposes.

4. Additional features
   - Add Wiener deconvolution, Butterworth filters, Laplacian sharpening, non-local means, bilateral filtering, and performance options (resize for quick previews).

## Quick examples and expectations

- Try "Power-Law Transformations" with gamma = 0.5 to brighten the image; set gamma = 2.0 to darken.
- Try "High-pass Filtering" to emphasize edges; expect halo/ringing if the radius is too sharp.
- Try "Homomorphic Filtering" to reduce uneven illumination; tune `cutoff`, `gamma_l`, and `gamma_h` in the code for better results.

## Closing summary

This README described every function in `main.py`, provided the math behind point, spatial, and frequency-domain operations, and gave instructions to run the Streamlit demo. For further help (e.g., adding unit tests or converting the geometric mean filter to a vectorized implementation), tell me what you'd like next and I will implement it.
