import os

# Define your directories
root_dir = 'D:/DomainAdap.Neural/Data/VKitti/VKITTI'
image_directory = os.path.join(root_dir, 'stylized clone')
label_directory = os.path.join(root_dir, 'sunset_labels')

# List all the images
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

# Loop through the image files
for image_file in image_files:
    # Replace the image file extension with the label file extension (assuming label files end with '.txt')
    # If your label files have a different naming pattern or extension, modify this part
    label_file = os.path.splitext(image_file)[0] + ".txt"  

    # Check if the label file exists
    if not os.path.exists(os.path.join(label_directory, label_file)):
        # If not, delete the image
        os.remove(os.path.join(image_directory, image_file))
        print(f"Deleted {image_file}, no matching label")
