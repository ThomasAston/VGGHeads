from head_detector import HeadDetector
# from deepface import DeepFace
import cv2
import os

# Initialize the detector
detector = HeadDetector()

# Specify the path to your image
image_path = "images/test.jpg"

# Get predictions
predictions = detector(image_path)

# Draw heads on the image
result_image = predictions.draw()
cv2.imwrite("result.png", result_image)

# Save head meshes
save_folder = "head_meshes"
os.makedirs(save_folder, exist_ok=True)
predictions.save_meshes(save_folder)

# Get and save aligned head crops
aligned_heads = predictions.get_aligned_heads()
for i, head in enumerate(aligned_heads):
    cv2.imwrite(f"aligned_head_{i}.png", head)

print(f"Detected {len(predictions.heads)} heads.")
print(f"Result image saved as 'result.png'")
print(f"Head meshes saved in '{save_folder}' folder")
print(f"Aligned head crops saved as 'aligned_head_*.png'")