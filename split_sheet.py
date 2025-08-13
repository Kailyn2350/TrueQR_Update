import os
import cv2

# --- Configuration ---
# This script will be run twice, once for True and once for False
# The source_dir and output_dir will be set dynamically in the main function
grid_layout = (5, 7) # 5 columns, 7 rows for many_QR.png

def split_image_sheet(image_path, output_folder, grid, base_filename):
    """Splits a single image sheet into a grid of smaller images."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    img_height, img_width, _ = img.shape
    cols, rows = grid
    
    cell_width = img_width // cols
    cell_height = img_height // rows

    img_count = 0
    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_width
            y1 = r * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            cropped_img = img[y1:y2, x1:x2]
            
            # Save the cropped image
            output_filename = f"{base_filename}_row{r}_col{c}.png"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, cropped_img)
            img_count += 1
            
    print(f"Split {image_path} into {img_count} images.")

def main():
    # Process for True images
    source_dir_true = os.path.join('sample', 'True')
    output_dir_true = os.path.join('Data', 'true_split')
    many_qr_path_true = os.path.join(source_dir_true, 'many_QR.png')

    if os.path.exists(many_qr_path_true):
        print(f"Processing {many_qr_path_true}...")
        split_image_sheet(many_qr_path_true, output_dir_true, grid_layout, 'many_QR_true')
    else:
        print(f"Warning: {many_qr_path_true} not found.")

    # Process for False images
    source_dir_false = os.path.join('sample', 'False')
    output_dir_false = os.path.join('Data', 'false_split')
    many_qr_path_false = os.path.join(source_dir_false, 'many_QR.png')

    if os.path.exists(many_qr_path_false):
        print(f"Processing {many_qr_path_false}...")
        split_image_sheet(many_qr_path_false, output_dir_false, grid_layout, 'many_QR_false')
    else:
        print(f"Warning: {many_qr_path_false} not found.")

    print("\nProcessing complete.")

if __name__ == '__main__':
    main()
