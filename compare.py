import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compare_images(original_path, reconstructed_path):
    # Load images
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(reconstructed_path)

    if img1 is None or img2 is None:
        print("Error: Could not load one of the images.")
        return

    # Ensure images are the same size (essential for pixel-wise metrics)
    if img1.shape != img2.shape:
        print(f"Resizing reconstructed image from {img2.shape[:2]} to {img1.shape[:2]}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 1. MSE (Mean Squared Error)
    # Lower is better. Measures the average squared difference between pixels.
    mse_value = np.mean((img1 - img2) ** 2)

    # 2. PSNR (Peak Signal-to-Noise Ratio)
    # Higher is better (usually 30-50dB). Measures ratio between max signal and noise.
    psnr_value = psnr(img1, img2)

    # 3. SSIM (Structural Similarity Index)
    # Range [0, 1]. Higher is better. 1.0 means identical.
    # We specify multichannel=True for RGB images.
    ssim_value = ssim(img1, img2, channel_axis=2)

    print("-" * 30)
    print(f"METRICS REPORT")
    print("-" * 30)
    print(f"MSE:  {mse_value:.4f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print("-" * 30)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def plot_codec_comparison(original_path, reconstructed_path, save_path='codec_report.png'):
    # 1. Load Images
    orig = cv2.imread(original_path)
    recon = cv2.imread(reconstructed_path)
    
    if orig is None or recon is None:
        raise FileNotFoundError("Check your image paths.")

    # Resize recon to match original if necessary
    if orig.shape != recon.shape:
        recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

    # 2. Calculate Scientific Metrics
    mse_val = np.mean((orig.astype(float) - recon.astype(float)) ** 2)
    psnr_val = psnr(orig, recon)
    ssim_val = ssim(orig, recon, channel_axis=2)

    # 3. Generate Difference Map (Heatmap)
    # Absolute difference boosted for visibility
    diff = cv2.absdiff(orig, recon)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Apply a JET colormap to make small errors look "hot"
    diff_heatmap = cv2.applyColorMap(cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)

    # 4. Create the Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Original
    axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=15)
    axes[0].axis('off')

    # Reconstructed
    axes[1].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"SIC Reconstruction", fontsize=15)
    axes[1].axis('off')

    # Difference Heatmap
    axes[2].imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Error Distribution (Heatmap)", fontsize=15)
    axes[2].axis('off')

    # Add Metrics Footer
    report = f"Metrics | MSE: {mse_val:.2f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
    plt.suptitle(report, fontsize=18, fontweight='bold', y=0.08)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    #plt.savefig(save_path)
    #print(f"Report saved to {save_path}")
    plt.show()

# Usage

"""plot_codec_comparison(
    r"C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\encoder\\data\\images\\Lenna.png",
    r"C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif_reconstructed.png" )
"""
compare_images(
    r"C:\\Users\\rical\\OneDrive\\Desktop\\Wallpaper\\Napoli.png",
    #r"C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\encoder\\data\\images\\Lenna.png",
    r"C:\\Users\\rical\\OneDrive\\Desktop\\Spatial_Inference_Codec\\build\\output_claude.sif_reconstructed.png" )