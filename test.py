import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
from model import UNET
from utils import load_checkpoint
import argparse
import numpy as np
import cv2

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 560
IMAGE_WIDTH = 690

# Function to overlay masks
def overlay_mask(image, mask, alpha=0.5):
    if isinstance(image, Image.Image):  # Convert PIL Image to NumPy array
        image = np.array(image)
    
    if len(mask.shape) == 2:  # Ensure mask is in correct format
        mask = np.expand_dims(mask, axis=-1) * 255  # Scale mask to [0, 255]

    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[:, :, 1] = mask.squeeze()  # Apply mask to green channel

    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay

# Function to load and preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Function to display images for test mode
def display_images(original_image, mask, overlay):
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

# Function to process all images in a folder
def process_folder(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = [os.path.join(input_folder, img) for img in os.listdir(input_folder)]
    for img_path in image_paths:
        # Preprocess image
        image_tensor = preprocess_image(img_path).to(DEVICE)

        # Predict mask
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(image_tensor))
            pred_mask = (pred_mask > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()

        # Load original image
        original_image = Image.open(img_path).convert("RGB")
        original_image_resized = original_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        # Create overlay
        overlayed_image = overlay_mask(np.array(original_image_resized), pred_mask)

        # Save overlayed image
        save_path = os.path.join(output_folder, os.path.basename(img_path))
        Image.fromarray(overlayed_image).save(save_path)

# Main function
def main(args):
    # Load model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(args.model_checkpoint, weights_only=False), model)
    model.eval()

    if args.mode == "test":
        # Single test image
        image_tensor = preprocess_image(args.test_image).to(DEVICE)
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(image_tensor))
            pred_mask = (pred_mask > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()

        original_image = Image.open(args.test_image).convert("RGB")
        original_image_resized = original_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        overlayed_image = overlay_mask(np.array(original_image_resized), pred_mask)

        # Display results
        display_images(original_image_resized, pred_mask, overlayed_image)
    elif args.mode == "folder":
        # Process all images in a folder
        process_folder(args.input_folder, args.output_folder, model)
    else:
        print("Invalid mode! Use 'test' or 'folder'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with UNET model for segmentation.")
    parser.add_argument("--mode", type=str, required=True, help="'test' for single image or 'folder' for batch processing.")
    parser.add_argument("--test_image", type=str, help="Path to the test image for 'test' mode.")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder for 'folder' mode.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder for 'folder' mode.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    args = parser.parse_args()

    main(args)
