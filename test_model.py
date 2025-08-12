import os
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET  # Ensure this imports your U-Net model

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define transformations (must match training transformations)
IMAGE_HEIGHT = 560
IMAGE_WIDTH = 690

transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

def load_model(checkpoint_path):
    """Load the trained U-Net model."""
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess an input image."""
    image = cv2.imread(image_path)
    augmented = transform(image=image)
    tensor_image = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor_image, image

def postprocess_output(output, original_shape):
    """Postprocess the model output into a binary mask."""
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    mask = output.squeeze(0).squeeze(0).cpu().numpy()
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    return (mask * 255).astype(np.uint8)

def display_image_with_mask(window_name, image, mask):
    """Update the OpenCV window with the current image and mask."""
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    cv2.imshow(window_name, overlay)

if __name__ == "__main__":
    # Path to the trained model
    checkpoint_path = "./checkpoints/UltimaVersion/my_checkpoint.pth.tar"

    # Load the trained model
    model = load_model(checkpoint_path)

    # Directory containing the input images
    input_dir = "../Riñón/Pruebas"
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg", "JPG"))]

    if not image_files:
        print("No images found in the directory!")
        exit()

    # Preload images and tensors
    preloaded_data = []
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        tensor_image, original_image = preprocess_image(image_path)
        preloaded_data.append((filename, tensor_image, original_image))

    # Window setup
    window_name = "Segmentation Viewer"
    cv2.namedWindow(window_name)

    # Loop to display images
    current_index = 0
    while True:
        # Get current data
        filename, tensor_image, original_image = preloaded_data[current_index]

        # Perform inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(tensor_image)

        # Postprocess the output
        mask = postprocess_output(output, original_image.shape[:2])

        # Display the image and mask
        display_image_with_mask(window_name, original_image, mask)

        # Key bindings
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # Next image
            current_index = (current_index + 1) % len(preloaded_data)
        elif key == ord('p'):  # Previous image
            current_index = (current_index - 1) % len(preloaded_data)
        elif key == ord('q'):  # Quit
            print("Exiting program...")
            cv2.destroyAllWindows()
            break
