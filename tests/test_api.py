import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("Health:", requests.get(f"{BASE_URL}/health").json())

def test_features():
    print("Features:", requests.get(f"{BASE_URL}/features").json())

def test_model_info():
    print("Model Info:", requests.get(f"{BASE_URL}/model_info").json())

def test_predict():
    payload = {
        "data": [
            [
                1.0, 9.488036, 0.000028, -0.000028, 170.53875, 0.00216, -0.00216,
                0.146, 0.318, -0.146, 2.9575, 0.0819, -0.0819, 616.0, 19.5, -19.5,
                2.26, 0.26, -0.15, 793.0, 93.59, 29.45, -16.65, 35.8, 1.0, 5455.0,
                81.0, -81.0, 4.467, 0.064, -0.096, 0.927, 0.105, -0.061, 291.93423,
                48.141651, 15.347
            ]
        ]
    }
    res = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Predict:", res.status_code, res.json())

if __name__ == "__main__":
    test_health()
    test_features()
    test_model_info()
    test_predict()