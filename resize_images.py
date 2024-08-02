from __future__ import print_function
from PIL import Image
import os

# Function to resize images to a target size while preserving aspect ratio
def resize_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size, Image.ANTIALIAS)
    return img

# Path to the folder containing content images
content_folder = 'content_images'
output_folder = 'resized_content_images'

# Resize content images to the target size and save them
for filename in os.listdir(content_folder):
    if filename.endswith(".jpg"):  # Assuming all content images are in JPG format
        # Construct the full path to the content image
        content_path = os.path.join(content_folder, filename)
        
        # Resize the content image to 224x224 while preserving aspect ratio
        resized_content_image = resize_image(content_path, (384, 384))
        
        # Save the resized content image
        resized_content_path = os.path.join(output_folder, "resized_" + filename)
        resized_content_image.save(resized_content_path)
        print("Resized content image saved:", resized_content_path)
