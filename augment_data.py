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

def random_blur(image):
    kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def random_perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    
    # Generate random offsets for each corner
    max_offset = int(min(h, w) * 0.1) # Max offset is 10% of the smallest dimension
    offsets = np.random.randint(-max_offset, max_offset, (4, 2)).astype(np.float32)
    pts2 = pts1 + offsets

    # Ensure the new points are within the image boundaries
    pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
    pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

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
        original_img = cv2.imread(img_path)

        if original_img is None:
            print(f"Could not read image: {img_path}")
            continue

        # Save original image to augmented folder as well
        cv2.imwrite(os.path.join(output_dir, filename), original_img)

        for i in range(num_augmentations):
            augmented_img = original_img.copy()

            # Apply a random sequence of augmentations
            if random.random() > 0.5:
                augmented_img = adjust_brightness(augmented_img, random.uniform(0.7, 1.3))
            
            if random.random() > 0.5:
                augmented_img = add_gaussian_noise(augmented_img, std_dev=random.uniform(5, 30))

            if random.random() > 0.5:
                augmented_img = random_blur(augmented_img)

            if random.random() > 0.5:
                augmented_img = random_perspective_transform(augmented_img)

            output_filename = f"{os.path.splitext(filename)[0]}_aug{i}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, augmented_img)
            
    print(f"Finished augmenting images from {source_dir}.")

def main():
    # Augment True images
    augment_and_save(os.path.join('Data', 'True'), os.path.join('Data', 'train_data_true'))
    
    # Augment False images
    augment_and_save(os.path.join('Data', 'False'), os.path.join('Data', 'train_data_false'))

    print("\nData augmentation complete.")

if __name__ == '__main__':
    main()