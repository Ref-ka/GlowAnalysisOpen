import os

from PIL import Image


# Function to crop an image from the center
def crop_image(input_image_path, output_image_path):
    try:
        # Open the input image
        with Image.open(input_image_path) as img:
            # Get the dimensions of the image
            img_width, img_height = img.size

            # Calculate the coordinates for the crop box
            left = img_width // 2 - 128
            upper = img_height // 2 - 128
            right = img_width // 2 + 128
            lower = img_height // 2 + 128

            # Crop the image using the calculated crop box
            cropped_image = img.crop((left, upper, right, lower))

            # Save the cropped image to the output path
            cropped_image.save(output_image_path)
            print(f"Cropped image saved to {output_image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    for path in list(map(lambda x: "21_01_25/" + x, os.listdir("21_01_25")[3:6])):
        crop_image(path, "cropped_images/" + path.split("/")[1])
