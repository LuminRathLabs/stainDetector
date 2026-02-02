# stainDetector
Industrial computer vision application for real time steel surface inspection using YOLO segmentation, RTSP video streams, and PLC integration. Designed for production environments with GPU acceleration and low latency processing.

# StainDetector - Industrial Stain Detection System

**StainDetector** is an advanced computer vision application designed for real-time detection of surface defects (stains) in industrial environments. It leverages YOLO-based deep learning models to identify anomalies on video streams (RTSP) and integrates with PLCs (Programmable Logic Controllers) for automated quality control.

## 🚀 Features

- **Real-Time Detection**: Processes live RTSP video streams to detect stains, welds (`Soldadura`), and strip ends (`Extremo`).
- **Deep Learning Powered**: Uses **YOLOv8** (Ultralytics) for high-accuracy object detection.
- **PLC Integration**: Communicates with Siemens PLCs via **Snap7** to trigger alarms or stop production lines upon defect detection.
- **Surface Sectorization**: Divides the inspection area into sectors for precise localization of defects.
- **GPU Acceleration**: optimized for NVIDIA GPUs using CUDA for low-latency performance.
- **Dashboard & Analytics**: Real-time visualization of detection statistics, frame rate, and system status.
- **Customizable**: Configurable detection zones, sensitivity thresholds, and model parameters.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**
- **NVIDIA GPU** (Recommended) with CUDA 11.8+ for real-time performance.
- **Visual C++ Build Tools** (Required for Snap7 on Windows).
- **FFmpeg**: The application includes a bundled FFmpeg binary in `bin/ffmpeg`, but ensure your system can run it.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/StainDetector.git
    cd StainDetector
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    *(Note: A `requirements.txt` is not provided, but here are the core packages)*
    ```bash
    pip install ultralytics opencv-python pillow numpy snap7 psutil
    ```

4.  **Install PyTorch with CUDA support**:
    To enable GPU acceleration, install the specific version of PyTorch compatible with your CUDA version (e.g., CUDA 11.8 or 12.1):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## 🖥️ Usage

### Running the Detector
To start the main detection interface:

```bash
python app/init.py
```

### Configuration
- **Model**: Select your YOLO model (`.pt`) file via the GUI.
- **Video Source**: Input the RTSP URL of your camera.
- **PLC Settings**: Configure the IP and Rack/Slot for your PLC in the `SendToPLC` settings window.

## 📂 Project Structure

- `app/`: Contains the main application source code.
  - `detect_manchas_gui_rtsp.py`: Main GUI application.
  - `sendToPLC_service.py`: Service for handling PLC communication.
  - `trainer_gui.py`: Utility for training new YOLO models.
- `bin/`: External binaries and libraries (FFmpeg, Snap7 DLLs).
- `config/`: Configuration files for the application.
- `models/`: Directory to store trained YOLO models (`.pt`).
- `docs/`: Documentation files.


