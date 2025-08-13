import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# New threshold based on user feedback, set between the two observed energy values
THRESHOLD = 5500.0

def analyze_image_frequency(image_path):
    """
    Analyzes an image's frequency domain. The logic is now INVERTED:
    a lower energy suggests a moire pattern is present.
    """
    # Read image in grayscale
    img = cv2.imread(image_path, 0)
    if img is None:
        return "Error", 0

    # Perform 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # --- Define Region of Interest (ROI) ---
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    mask = np.ones_like(magnitude_spectrum, dtype=bool)
    mask[crow-30:crow+30, ccol-30:ccol+30] = False
    mask[crow-5:crow+5, :] = False
    mask[:, ccol-5:ccol+5] = False

    # Calculate the mean magnitude in the ROI
    roi_energy = np.mean(magnitude_spectrum[mask])

    # --- Prediction (INVERTED LOGIC) ---
    # If energy is LESS than the threshold, we predict it's a TRUE image.
    prediction = "True" if roi_energy < THRESHOLD else "False"
    
    return prediction, roi_energy

def main():
    # --- File Selection ---
    true_path = os.path.join('Data', 'true')
    false_path = os.path.join('Data', 'false')

    if not os.path.exists(true_path) or not os.path.exists(false_path):
        print(f"Error: Make sure 'Data/true' and 'Data/false' directories exist.")
        return

    try:
        true_image_name = random.choice(os.listdir(true_path))
        false_image_name = random.choice(os.listdir(false_path))
    except IndexError:
        print("Error: Make sure there are images in 'Data/true' and 'Data/false' directories.")
        return

    true_image_path = os.path.join(true_path, true_image_name)
    false_image_path = os.path.join(false_path, false_image_name)

    # --- Analysis and Visualization ---
    plt.figure(figsize=(12, 7))

    # 1. Process and display the TRUE image
    prediction_true, energy_true = analyze_image_frequency(true_image_path)
    img_true = cv2.imread(true_image_path, 0)
    plt.subplot(1, 2, 1)
    plt.imshow(img_true, cmap='gray')
    title_true = (
        f"Ground Truth: True\nPrediction: {prediction_true}\n"
        f"Energy: {energy_true:.1f} / Threshold: {THRESHOLD:.1f}"
    )
    plt.title(title_true)
    plt.xticks([]), plt.yticks([])

    # 2. Process and display the FALSE image
    prediction_false, energy_false = analyze_image_frequency(false_image_path)
    img_false = cv2.imread(false_image_path, 0)
    plt.subplot(1, 2, 2)
    plt.imshow(img_false, cmap='gray')
    title_false = (
        f"Ground Truth: False\nPrediction: {prediction_false}\n"
        f"Energy: {energy_false:.1f} / Threshold: {THRESHOLD:.1f}"
    )
    plt.title(title_false)
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.savefig('verification_result_final.png')
    print("Final verification result saved to verification_result_final.png")

if __name__ == '__main__':
    main()