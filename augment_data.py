
import os
import cv2
import numpy as np
import random

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, factor)
    v = np.clip(v, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def add_gaussian_noise(image, mean=0, std_dev=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def augment_and_save(source_dir, output_dir, num_augmentations=9):
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Augmenting {len(image_files)} images from {source_dir}...")

    for filename in image_files:
        img_path = os.path.join(source_dir, filename)
        original_img = cv2.imread(img_path) # Read in BGR for color adjustments

        if original_img is None:
            print(f"Could not read image: {img_path}")
            continue

        # Save original image to augmented folder as well
        cv2.imwrite(os.path.join(output_dir, filename), original_img)

        for i in range(num_augmentations):
            # Random brightness factor (0.8 to 1.2)
            brightness_factor = random.uniform(0.8, 1.2)
            bright_img = adjust_brightness(original_img, brightness_factor)

            # Random noise std_dev (10 to 40)
            noise_std_dev = random.uniform(10, 40)
            noisy_img = add_gaussian_noise(bright_img, std_dev=noise_std_dev)

            output_filename = f"{os.path.splitext(filename)[0]}_aug{i}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, noisy_img)
    print(f"Finished augmenting images from {source_dir}.")

def main():
    # Augment True images
    augment_and_save(os.path.join('Data', 'true_split'), os.path.join('Data', 'true_augmented'))
    
    # Augment False images
    augment_and_save(os.path.join('Data', 'false_split'), os.path.join('Data', 'false_augmented'))

    print("\nData augmentation complete.")

if __name__ == '__main__':
    main()
