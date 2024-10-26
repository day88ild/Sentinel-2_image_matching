# Satellite Image Matching and Keypoint Visualization

This project aims to perform image matching and keypoint visualization for satellite images, specifically using Sentinel-2 imagery. The algorithm can compare two images and determine if they match based on detected keypoints. It also provides a visual representation of the matching keypoints, making it suitable for analyzing satellite images across different seasons or conditions.

## Project Structure

Here's an overview of the files in the repository:

- `inference.py`: Script for running inference.
- `Data_preporation.ipynb`: Jupyter notebook for preprocessing the dataset.
- `README.md`: This file, providing an overview and instructions for the project.
- `requirements.txt`: List of dependencies required to run the project.
- `Report.pdf`: This file provides report with potentioal strategies to use.
- `demo.ipynb`: Demo
- `images_to_choose`: contains images for inference.


## Getting Started

### Prerequisites

To run the project, you will need:

- Python 3.8 or later
- Required libraries listed in `requirements.txt`

### Installation

1. **Clone the Repository:**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1. **Run Inference:**

   You can use the `inference.py` script for a command-line based approach (it works on already prepared images):

   ```bash
   python inference.py --image1 path/to/first/image --image2 path/to/second/image
   ```
