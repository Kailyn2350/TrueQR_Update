
import cv2
import numpy as np
import os

def generate_negative_samples(output_dir, num_images=50, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Generate blank images (black, white, gray)
    cv2.imwrite(os.path.join(output_dir, "neg_blank_black.png"), np.zeros(size, dtype=np.uint8))
    cv2.imwrite(os.path.join(output_dir, "neg_blank_white.png"), np.ones(size, dtype=np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, "neg_blank_gray.png"), np.ones(size, dtype=np.uint8) * 128)

    # 2. Generate images with random noise
    for i in range(num_images):
        noise_img = np.random.randint(0, 256, size, dtype=np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"neg_noise_{i:03d}.png"), noise_img)

    print(f"Generated {num_images + 3} negative samples in {output_dir}")

if __name__ == "__main__":
    output_directory = os.path.join("Data", "False")
    generate_negative_samples(output_directory)
