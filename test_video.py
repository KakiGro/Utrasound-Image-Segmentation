import cv2
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0
print(torch.cuda.get_device_name(0))  # Should print your GPU name
import numpy as np
from model import UNET
from utils import load_checkpoint
from PIL import Image
import torchvision.transforms as transforms
import time  # For measuring processing times

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 690
IMAGE_WIDTH = 560

def overlay_mask(image, mask, alpha=0.5):
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[:, :, 1] = mask  # Apply mask to green channel
    return cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)

def preprocess_frame(frame):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    return transform(frame_pil).unsqueeze(0).to(DEVICE)

def main(video_path, checkpoint_path):
    print(f"Using device: {DEVICE}")
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    load_checkpoint(checkpoint, model)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = preprocess_frame(frame).to(DEVICE)
        print(f"Tensor device: {input_tensor.device}")
        print(f"Model device: {next(model.parameters()).device}")

        
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(input_tensor))
            pred_mask = (pred_mask > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()
        
        mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0])).astype(np.uint8) * 255
        overlayed_frame = overlay_mask(frame, mask_resized)
        combined_frame = np.hstack((frame, overlayed_frame))
        
        cv2.imshow("Original and Masked", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main("./KidneyUltrasound.mp4", "./Checkpoints/UltimaVersion/my_checkpoint.pth.tar")
