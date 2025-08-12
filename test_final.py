import cv2
import torch
import numpy as np
from model import UNET
from utils import load_checkpoint
import time

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 690
IMAGE_WIDTH = 560

# Preprocess the frame for the model
def preprocess_frame(frame, device, input_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    frame_resized = cv2.resize(frame, input_size)
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    return frame_tensor

# Post-process the model's output
def postprocess_mask(mask, original_size):
    mask = mask.squeeze().cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  # Apply threshold
    mask_resized = cv2.resize(mask, original_size)
    return mask_resized

def main(video_path, checkpoint_path):
    # Load the model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False), model)
    model.eval()

    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame to select the ROI
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Allow user to select ROI
    print("Select the region of interest (ROI) and press Enter.")
    roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    print(f"Selected ROI: {roi}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()  # Measure time for debugging

        # Crop the frame to the selected ROI
        x, y, w, h = map(int, roi)
        cropped_frame = frame[y:y+h, x:x+w]

        # Preprocess the cropped frame for the model
        input_tensor = preprocess_frame(cropped_frame, DEVICE)

        # Get the model's prediction
        with torch.no_grad():
            output = model(input_tensor)
            mask = torch.sigmoid(output).squeeze(0)

        # Debug: Print mask statistics
        print(f"Mask values: min={mask.min().item()}, max={mask.max().item()}")

        # Post-process the mask
        mask_binary = postprocess_mask(mask, (w, h))

        # Stack the cropped frame and the binary mask side by side
        mask_colored = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack((cropped_frame, mask_colored))

        # Display the combined view
        cv2.imshow('Original and Segmentation Mask', combined_view)

        # Debug: Measure processing time
        print(f"Frame processed in {time.time() - start_time:.2f} seconds")

        # Add a delay to ensure enough time for visualization
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the function
main("./KidneyUltrasound.mp4", "./Checkpoints/UltimaVersion/my_checkpoint.pth.tar")
