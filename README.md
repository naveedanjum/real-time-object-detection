# Real Time Object Detection App

This project is a FastAPI-based web application that performs real-time object detection using the YOLOv8 model from a camera stream. Detected objects are saved with a timestamp in a JSON file, and annotated images are stored in an organized directory structure.

## Features

- Real-time object detection using YOLOv8
- Save detections with a UTC timestamp in a daily JSON file
- Save annotated frames as images
- Organized project structure following best OOP practices

## Project Structure
object_detection_app/
├── app/
│ ├── init.py
│ ├── main.py
│ ├── api/
│ │ ├── init.py
│ │ ├── endpoints.py
│ ├── core/
│ │ ├── init.py
│ │ ├── config.py
│ ├── services/
│ │ ├── init.py
│ │ ├── camera.py
│ │ ├── detection.py
│ ├── utils/
│ │ ├── init.py
│ │ ├── file_utils.py
└── run.py


## Getting Started

### Prerequisites

- Python 3.7+
- Pip package manager

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/object_detection_app.git
    cd object_detection_app
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To start the application, run:

```bash
python run.py


