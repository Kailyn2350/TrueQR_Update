import os
import cv2
import numpy as np

def process_images_for_fft(source_dir, output_dir):
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing {len(image_files)} images from {source_dir}...")

    for filename in image_files:
        img_path = os.path.join(source_dir, filename)
        img = cv2.imread(img_path, 0) # Read in grayscale

        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        # Perform FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1) # Add 1 to avoid log(0)

        # Normalize to 0-255 for saving as image
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = np.uint8(magnitude_spectrum)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, magnitude_spectrum)
    print(f"Finished processing images for FFT from {source_dir}.")

def main():
    # Process True images from augmented data
    process_images_for_fft(os.path.join('Data', 'true_augmented'), os.path.join('Data', 'fft_true'))
    
    # Process False images from augmented data
    process_images_for_fft(os.path.join('Data', 'false_augmented'), os.path.join('Data', 'fft_false'))

    print("\nFFT preprocessing complete.")

if __name__ == '__main__':
    main()