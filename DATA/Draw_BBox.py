import cv2

# Load the image
image_path = "D:/VKitti/vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00001.jpg"
image = cv2.imread(image_path)

# Read the label file
label_path = "D:/VKitti/VKITTI/VKITTI_ANNOTATIONS/Scene01 clone (0, 0).txt"
with open(label_path, 'r') as label_file:
    labels = label_file.readlines()

# Define the class names (corresponding to class indices)
class_names = [0, "Car", "Van", "Truck",]  # Modify this list based on your dataset

# Loop through each line in the label file
for label in labels:
    label = label.strip().split()
    class_index = int(label[0])
    x_normalized, y_normalized, width_normalized, height_normalized = map(float, label[1:5])

    # Convert normalized coordinates to absolute coordinates
    image_height, image_width, _ = image.shape
    x = int(x_normalized * image_width)
    y = int(y_normalized * image_height)
    width = int(width_normalized * image_width)
    height = int(height_normalized * image_height)

    # Draw the bounding box on the image
    color = (0, 255, 0)  # You can change the color if needed
    thickness = 2  # You can adjust the thickness of the bounding box
    cv2.rectangle(image, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2), color, thickness)
    cv2.putText(image, class_names[class_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)

# Display the image with bounding boxes
cv2.imshow("Image with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()