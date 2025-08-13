
import os
from PIL import Image, ImageDraw

# --- Configuration ---
source_dir = 'moire_images_light'
output_pdf = 'output_grid_padded_a4.pdf'
grid_layout = (5, 7) # 5 columns, 7 rows
cell_padding = 40 # pixels of padding around each image inside its cell

# A4 dimensions in pixels at 300 DPI
A4_WIDTH = 2480
A4_HEIGHT = 3508

# --- Get Image List ---
image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
if not image_files:
    print(f"No images found in '{source_dir}'.")
    exit()

# --- PDF Creation ---
# Create a new white A4 canvas for the single page
canvas = Image.new('RGB', (A4_WIDTH, A4_HEIGHT), 'white')

# --- Calculate image placement and size ---
cell_width = A4_WIDTH // grid_layout[0]
cell_height = A4_HEIGHT // grid_layout[1]

if len(image_files) > grid_layout[0] * grid_layout[1]:
    print(f"Warning: More images than can fit in the grid. Only the first {grid_layout[0] * grid_layout[1]} will be included.")

for i, filename in enumerate(image_files):
    if i >= grid_layout[0] * grid_layout[1]:
        break # Stop if we run out of grid space

    img_path = os.path.join(source_dir, filename)
    img = Image.open(img_path).convert('RGB')

    # Resize image to fit inside the padded cell, maintaining aspect ratio
    thumbnail_size = (cell_width - cell_padding, cell_height - cell_padding)
    img.thumbnail(thumbnail_size)

    # Create a new blank cell canvas
    cell_canvas = Image.new('RGB', (cell_width, cell_height), 'white')
    
    # Paste the resized image into the center of the cell canvas
    paste_x = (cell_width - img.width) // 2
    paste_y = (cell_height - img.height) // 2
    cell_canvas.paste(img, (paste_x, paste_y))
    
    # Add a border to the cell for easier cutting
    draw = ImageDraw.Draw(cell_canvas)
    draw.rectangle([0, 0, cell_width-1, cell_height-1], outline="gray")

    # Calculate position in grid
    row = i // grid_layout[0]
    col = i % grid_layout[0]

    # Paste the cell canvas onto the main A4 canvas
    final_paste_x = col * cell_width
    final_paste_y = row * cell_height
    canvas.paste(cell_canvas, (final_paste_x, final_paste_y))

# Save the canvas as a single-page PDF
canvas.save(output_pdf)

print(f"Successfully created '{output_pdf}' with a padded grid of {len(image_files)} images.")

