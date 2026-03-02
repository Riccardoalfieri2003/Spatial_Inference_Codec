import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_lab(image_path):
    # 1. Load the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Could not find image.")
        return

    # 2. Convert BGR to RGB (for Matplotlib) and LAB (for processing)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    # 3. Split the channels
    # L = Lightness (0-100 in theory, 0-255 in OpenCV uint8)
    # a = Green-Red axis
    # b = Blue-Yellow axis
    L, a, b = cv2.split(img_lab)

    # 4. Create the plot
    plt.figure(figsize=(15, 8))

    # Original RGB
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Original (RGB)')
    plt.axis('off')

    # L Channel (Shading/Luminance)
    plt.subplot(1, 4, 2)
    plt.imshow(L, cmap='gray')
    plt.title('L Channel (Lightness)')
    plt.axis('off')

    # a Channel (Green to Red)
    plt.subplot(1, 4, 3)
    plt.imshow(a, cmap='RdYlGn_r') # Red-Yellow-Green map to see the axis
    plt.title('a Channel (G-R)')
    plt.axis('off')

    # b Channel (Blue to Yellow)
    plt.subplot(1, 4, 4)
    plt.imshow(b, cmap='YlGnBu_r') # Yellow-Green-Blue map
    plt.title('b Channel (B-Y)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_lab_differences(orig_path, recon_path):
    # 1. Load images
    orig_bgr = cv2.imread(orig_path)
    recon_bgr = cv2.imread(recon_path)

    if orig_bgr is None or recon_bgr is None:
        print("Error: Check your file paths.")
        return

    # Ensure they are the same size
    if orig_bgr.shape != recon_bgr.shape:
        print(f"Resizing reconstructed image from {recon_bgr.shape[:2]} to {orig_bgr.shape[:2]}")
        recon_bgr = cv2.resize(recon_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

    # 2. Convert to LAB
    orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    recon_lab = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    # 3. Calculate Absolute Difference per channel
    # Using float32 prevents 'wrap-around' errors during subtraction
    diff_L = np.abs(orig_lab[:,:,0] - recon_lab[:,:,0])
    diff_a = np.abs(orig_lab[:,:,1] - recon_lab[:,:,1])
    diff_b = np.abs(orig_lab[:,:,2] - recon_lab[:,:,2])

    # 4. Calculate Stats (MSE per channel)
    mse_L = np.mean(diff_L**2)
    mse_a = np.mean(diff_a**2)
    mse_b = np.mean(diff_b**2)

    print(f"--- Channel Error Report (MSE) ---")
    print(f"L-Channel (Shading): {mse_L:.4f}")
    print(f"a-Channel (Green-Red): {mse_a:.4f}")
    print(f"b-Channel (Blue-Yellow): {mse_b:.4f}")

    # 5. Visualize
    plt.figure(figsize=(18, 6))

    # L-Diff
    plt.subplot(1, 3, 1)
    # 'inferno' makes small errors (1-5 units) visible as orange/red
    plt.imshow(diff_L, cmap='inferno', vmin=0, vmax=20) 
    plt.colorbar(label='Error Magnitude')
    plt.title(f'L-Channel Error\n(MSE: {mse_L:.2f})')
    plt.axis('off')

    # a-Diff
    plt.subplot(1, 3, 2)
    plt.imshow(diff_a, cmap='inferno', vmin=0, vmax=20)
    plt.colorbar(label='Error Magnitude')
    plt.title(f'a-Channel Error\n(MSE: {mse_a:.2f})')
    plt.axis('off')

    # b-Diff
    plt.subplot(1, 3, 3)
    plt.imshow(diff_b, cmap='inferno', vmin=0, vmax=20)
    plt.colorbar(label='Error Magnitude')
    plt.title(f'b-Channel Error\n(MSE: {mse_b:.2f})')
    plt.axis('off')

    plt.suptitle("Geometric Error Heatmap (Brighter = More Difference)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Run it
orig = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\encoder\\data\\images\\Lenna.png"
#recon = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\out_full_reconstruction.png"

recon = "C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\out_per_channel_reconstruction.png"
compare_lab_differences(orig, recon)

