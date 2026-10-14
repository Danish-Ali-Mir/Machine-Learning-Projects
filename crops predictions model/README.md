# 🌾 Crop Recommendation System

A machine learning web app that recommends the most suitable crop to grow based on soil nutrients and local weather conditions. It uses a Random Forest classifier trained on 2,200 records, a FastAPI backend, and a simple HTML front end.

## Features

- Predicts the best crop from **22 crop types** (rice, maize, coffee, apple, mango, and more)
- Returns a **confidence score** and the **top 3 candidate crops**
- Clean web interface for entering soil and weather values
- REST API with automatic input validation
- Interactive API documentation at `/docs`
- About **99% accuracy** on the held-out test set

## Tech Stack

| Part | Technology |
|---|---|
| Machine learning | scikit-learn (Random Forest Classifier) |
| Backend / API | FastAPI + Uvicorn |
| Front end | HTML, CSS, JavaScript |
| Data handling | pandas, joblib |

## Project Structure

```
crop-recommendation/
├── template/
│   └── index.html            # Web page (front end)
├── main.py                   # FastAPI app: serves the page and the /predict API
├── train_model.py            # Trains the model and saves it to crop_model.joblib
├── crop_model.joblib         # Saved trained model
├── Crop_recommendation.csv   # Dataset
├── file.ipynb                # Original exploration notebook
├── requirements.txt          # Python dependencies
└── README.md
```

## Dataset

`Crop_recommendation.csv` contains 2,200 rows and 22 crops (100 samples per crop).

| Feature | Description |
|---|---|
| `N` | Nitrogen content in the soil |
| `P` | Phosphorus content in the soil |
| `K` | Potassium content in the soil |
| `temperature` | Temperature in °C |
| `humidity` | Relative humidity in % |
| `ph` | Soil pH value |
| `rainfall` | Rainfall in mm |
| `label` | Target: the recommended crop |

## Getting Started

### Prerequisites

- Python 3.9 or newer
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the model

This step creates `crop_model.joblib`. If the file is already in the repository you can skip it. Run it again only if you change the data or the model.

```bash
python train_model.py
```

### Run the app

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

Interactive API docs are available at **http://127.0.0.1:8000/docs**.

## Usage

### Web interface

1. Enter the soil values (N, P, K, pH) and weather values (temperature, humidity, rainfall).
2. Click **Recommend a crop**.
3. The best match, its confidence, and the other top candidates appear in the result panel.

Use **Fill in an example** to load sample values.

### API

**`POST /predict`**

Request body:

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.88,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202.9
}
```

Response:

```json
{
  "recommended_crop": "rice",
  "confidence": 0.99,
  "top_3": [
    { "crop": "rice", "probability": 0.99 },
    { "crop": "jute", "probability": 0.01 },
    { "crop": "pomegranate", "probability": 0.0 }
  ]
}
```

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.88,"humidity":82,"ph":6.5,"rainfall":202.9}'
```

Example with Python:

```python
import requests

data = {"N": 90, "P": 42, "K": 43, "temperature": 20.88,
        "humidity": 82, "ph": 6.5, "rainfall": 202.9}
print(requests.post("http://127.0.0.1:8000/predict", json=data).json())
```

Input validation: `ph` must be between 0 and 14, `humidity` between 0 and 100, and `N`, `P`, `K`, `rainfall` must be 0 or higher. Invalid input returns a `422` error.

## How It Works

```
Browser form  →  POST /predict  →  FastAPI (main.py)  →  Random Forest model
                                                              ↓
Result panel  ←  JSON response  ←  FastAPI (main.py)  ←  crop + probabilities
```

1. `train_model.py` trains a `RandomForestClassifier` on 80% of the data and saves it with `joblib`.
2. `main.py` loads the saved model once at startup.
3. The web page sends the form values to `/predict`, and the API returns the prediction.

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Train / test split | 80% / 20% (`random_state=42`) |
| Test accuracy | ~99.3% |

## Future Improvements

- Add crop images and growing tips to the results
- Compare other algorithms (XGBoost, SVM) and tune hyperparameters
- Show feature importance to explain each prediction
- Deploy online (Render, Railway, or similar)
- Add unit tests for the API

## License

This project is licensed under the MIT License. Add a `LICENSE` file if you want to use this license.

## Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username)
