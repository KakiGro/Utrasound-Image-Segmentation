# Kidney Ultrasound Segmentation with U-Net

A deep learning project for automatic kidney segmentation in ultrasound images using a U-Net architecture. This project includes training, testing, and real-time inference capabilities with both GUI and command-line interfaces.

## 🎯 Overview

This project implements a U-Net model for semantic segmentation of kidney structures in ultrasound images. It features:

- **U-Net Architecture**: Custom implementation with configurable features
- **Real-time Processing**: Live camera and video processing capabilities
- **GUI Application**: User-friendly interface for model testing and inference
- **Preprocessing Pipeline**: Histogram specification and ultrasound cone detection
- **Multiple Testing Modes**: Support for images, videos, and live camera feeds

## 📁 Project Structure

```
Modelo/
├── app.py                 # GUI application with Tkinter
├── model.py              # U-Net model architecture
├── train.py              # Training script
├── dataset.py            # Dataset loader for training
├── utils.py              # Utility functions (checkpoints, loaders, accuracy)
├── test*.py              # Various testing scripts
├── Checkpoints/          # Model checkpoints directory
├── data/                 # Training and validation data
└── media/                # Sample images and videos
```

## 🚀 Features

### Model Architecture
- **U-Net**: Encoder-decoder architecture with skip connections
- **Input Size**: 690x560 pixels
- **Channels**: 3 input channels (RGB), 1 output channel (mask)
- **Features**: [64, 128, 256, 512] feature maps

### Preprocessing
- Histogram specification for image enhancement
- Ultrasound cone detection and masking
- Automatic image resizing and normalization

### Applications
- **GUI Interface**: Interactive application for real-time processing
- **Batch Processing**: Process entire folders of images
- **Video Processing**: Frame-by-frame segmentation
- **Live Camera**: Real-time segmentation from camera feed

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install scikit-image
pip install tqdm
pip install numpy
pip install tkinter
```

## 📊 Usage

### Training
```bash
python train.py
```

Configure training parameters in `train.py`:
- Learning rate: 1e-4
- Batch size: 16
- Image dimensions: 690x560
- Data paths for training and validation

### GUI Application
```bash
python app.py
```

The GUI provides:
- Model checkpoint selection
- Processing mode selection (image/video/camera)
- Real-time visualization with mask overlay
- Preprocessing options

### Command Line Testing

#### Single Image Processing
```bash
python test.py --input path/to/image.jpg --checkpoint path/to/model.pth
```

#### Video Processing
```bash
python test_video.py path/to/video.mp4 path/to/checkpoint.pth
```

#### Live Camera
```bash
python test_camera.py 0 path/to/checkpoint.pth
```

#### Accuracy Evaluation
```bash
python test_accuracy.py
```

## 🎛️ Configuration

### Model Parameters
- **Input Channels**: 3 (RGB)
- **Output Channels**: 1 (Binary mask)
- **Image Height**: 560
- **Image Width**: 690
- **Device**: CUDA if available, else CPU

### Training Hyperparameters
- **Learning Rate**: 1e-4
- **Batch Size**: 16
- **Epochs**: 3 (configurable)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross Entropy

## 📈 Model Performance

The model achieves segmentation of kidney structures in ultrasound images with:
- Real-time inference capability
- Robust performance across different ultrasound qualities
- Preprocessing pipeline for enhanced accuracy

## 🗂️ Data Structure

Expected data organization:
```
data/
├── train_images/         # Training ultrasound images
├── train_masks/          # Corresponding segmentation masks
├── val_images/           # Validation images
└── val_masks/            # Validation masks
```

## 🔧 Key Components

### Model (`model.py`)
- `UNET`: Main U-Net architecture
- `DoubleConv`: Double convolution block with batch normalization

### Training (`train.py`)
- Mixed precision training with autocast
- Checkpoint saving and loading
- Progress tracking with tqdm

### Utilities (`utils.py`)
- `get_loaders()`: Data loader creation
- `check_accuracy()`: Model evaluation
- `save_predictions_as_imgs()`: Visualization of results

### GUI Application (`app.py`)
- Real-time processing interface
- Multiple input sources (image, video, camera)
- Preprocessing options and visualization

## 🎥 Testing Scripts

- `test.py`: Batch image processing
- `test_video.py`: Video file processing
- `test_camera.py`: Live camera processing
- `test_final.py`: Final model evaluation
- `test_accuracy.py`: Accuracy metrics calculation

## 📋 Requirements

- PyTorch >= 1.7.0
- OpenCV >= 4.0
- PIL (Pillow)
- scikit-image
- NumPy
- tqdm
- tkinter (usually included with Python)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- U-Net architecture based on the original paper by Ronneberger et al.
- Built with PyTorch framework
- GUI implementation using Tkinter

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This project is designed for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.