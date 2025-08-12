import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from model import UNET
from utils import load_checkpoint, get_loaders, save_predictions_as_imgs

# Hyperparameters and configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_HEIGHT = 560
IMAGE_WIDTH = 690
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "./data/train_images/"
TRAIN_MASK_DIR = "./data/train_masks/"
VAL_IMG_DIR = "./data/val_images/"
VAL_MASK_DIR = "./data/val_masks/"

def check_accuracy_stats(loader, model, device="cuda"):
    model.eval()
    accuracies = []
    dice_scores = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Add channel dimension to masks
            y = y.to(device).unsqueeze(1)
            
            # Forward pass and predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # Compute accuracy for this batch
            num_correct = (preds == y).sum().item()
            num_pixels = torch.numel(preds)
            batch_acc = num_correct / num_pixels
            accuracies.append(batch_acc)
            
            # Compute Dice score for this batch
            intersection = (preds * y).sum()
            dice = (2 * intersection) / ((preds + y).sum() + 1e-8)
            dice_scores.append(dice.item())
    
    model.train()
    
    # Calculate statistics using numpy
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"Accuracy: Mean = {mean_acc*100:.2f}%, Std = {std_acc*100:.2f}%")
    print(f"Dice Score: Mean = {mean_dice:.4f}, Std = {std_dice:.4f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test model accuracy and Dice score, with mean and standard deviation, and optionally save predictions."
    )
    parser.add_argument("--save_preds", action="store_true",
                        help="Set this flag to save prediction images.")
    parser.add_argument("--pred_folder", type=str, default="./saved_images/",
                        help="Folder where prediction images will be saved.")
    args = parser.parse_args()
    
    # Define the validation transforms
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    # Initialize the model and load checkpoint if available
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    if LOAD_MODEL:
        checkpoint = torch.load("./Checkpoints/UltimaVersion/my_checkpoint.pth.tar", map_location=DEVICE)
        load_checkpoint(checkpoint, model)
    
    # Prepare the validation data loader
    # Training directories are not needed so empty strings are passed.
    _, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=None,
        val_transform=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    
    # Evaluate and print the mean and standard deviation for accuracy and Dice score
    check_accuracy_stats(val_loader, model, device=DEVICE)
    
    # Optionally, save the prediction images to the specified folder
    if args.save_preds:
        print(f"Saving predictions to folder: {args.pred_folder}")
        save_predictions_as_imgs(val_loader, model, folder=args.pred_folder, device=DEVICE)

if __name__ == "__main__":
    main()
