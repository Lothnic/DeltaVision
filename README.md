# DeltaVision

A Python-based tool for detecting changes between two images using computer vision techniques.

Live on -> https://deltavision.streamlit.app/

## Features

- Image alignment using SIFT feature matching
- Change detection via absolute difference
- Heatmap visualization of changes
- Command-line interface for batch processing
- Web interface via Streamlit
- REST API via FastAPI

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Local Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DeltaVision.git
   cd DeltaVision
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -e .
   ```

## Usage

### Command Line
Process two images:
```
python main.py --image1 path/to/image1.tif --image2 path/to/image2.tif --output output.png --report report.json
```

Process a directory of images:
```
python main.py --input-dir path/to/images --output-dir output/
```

### Streamlit Web App
Run locally:
```
streamlit run streamlit_app.py
```

Access at http://localhost:8501

### API
Start the FastAPI server:
```
uvicorn main:app --reload
```

## Deployment

### Streamlit Cloud
1. Push code to a public GitHub repository
2. Go to share.streamlit.io
3. Connect your repo and deploy

### Other Platforms
- Heroku: Use the provided requirements.txt
- Docker: Build from the project files

## Project Structure

- `src/`: Core modules
  - `image_processing.py`: Image loading, resizing, and alignment
  - `change_detection.py`: Difference computation
  - `visualization.py`: Heatmap creation
- `main.py`: Command-line interface
- `streamlit_app.py`: Web interface
- `tests/`: Unit tests

## Configuration

- SIFT parameters can be adjusted in `src/image_processing.py`
- Change threshold in `main.py` (default: 30)

