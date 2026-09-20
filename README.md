# Predictive Maintenance & Machine Failure Prediction

An end-to-end machine learning system that predicts industrial machine failures from sensor data, deployed as a containerized API and demonstrated with a simulated IoT sensor stream over MQTT.

## Overview

This project goes beyond a typical notebook-based ML exercise: it trains a failure-prediction model on real industrial sensor data, properly handles severe class imbalance with cross-validated threshold selection, serves the model through a REST API, and simulates a live IoT sensor feed to demonstrate real-time inference.

**Pipeline:** `Sensor data → Feature engineering → XGBoost classifier → FastAPI serving → MQTT streaming simulation → Live dashboard`

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 synthetic industrial sensor readings modeled on real milling machine behavior, with a binary failure label and five failure-mode sub-labels (TWF, HDF, PWF, OSF, RNF). Only ~3.4% of records represent a failure, making class imbalance a central challenge.

**Features used:**
- Machine type (L / M / H, one-hot encoded)
- Air temperature [K], Process temperature [K]
- Rotational speed [rpm], Torque [Nm], Tool wear [min]
- Engineered: temperature difference, mechanical power

## Modeling approach

- **Model:** XGBoost classifier, tuned via `GridSearchCV`
- **Class imbalance:** addressed through threshold tuning rather than relying on the default 0.5 cutoff
- **Avoiding data leakage:** the decision threshold was initially selected on the test set, which produced an optimistic result. This was corrected by re-selecting the threshold via `cross_val_predict` on out-of-fold training predictions, then evaluating once on the held-out test set — the methodologically correct order

### Final results (held-out test set, threshold = 0.30)

| Metric | Score |
|---|---|
| Precision | 86.4% |
| Recall | 83.8% |
| F1 | 85.1% |
| Accuracy | 99.1% |

Threshold 0.30 was chosen because recall saturates below this point on the precision-recall curve — lowering the threshold further only hurts precision with no recall benefit, and in a maintenance context, missed failures are costlier than false alarms.

## Architecture

```
publisher.py  --->  Mosquitto (MQTT broker)  --->  subscriber.py / dashboard.py  --->  FastAPI /predict  --->  XGBoost model
(simulated        (topic: factory/machine1/       (independent consumers of        (inference API)
 sensor)            sensors)                        the same stream)
```

- **`publisher.py`** simulates an IoT sensor by replaying dataset rows over MQTT with a configurable delay, mimicking periodic sensor reporting
- **Mosquitto** is the MQTT broker — publishers and subscribers never talk to each other directly, only through named topics, which is the standard IoT pub/sub pattern
- **`subscriber.py`** is a terminal-based listener that forwards each reading to the API and logs the prediction
- **`dashboard.py`** is a Streamlit live dashboard showing incoming readings, a real-time failure-risk chart, and highlighted failure predictions
- **`main.py`** is the FastAPI service that loads the trained model and threshold, and exposes `/predict`, `/predict/batch`, and `/health` endpoints

## Tech stack

Python · scikit-learn · XGBoost · pandas · FastAPI · Pydantic · Docker & Docker Compose · Eclipse Mosquitto (MQTT) · paho-mqtt · Streamlit

## Project structure

```
.
├── main.py                  # FastAPI serving layer
├── publisher.py              # Simulated IoT sensor (MQTT publisher)
├── subscriber.py             # Terminal MQTT subscriber + inference bridge
├── dashboard.py               # Streamlit live dashboard (MQTT subscriber)
├── docker-compose.yml         # Mosquitto broker + API service
├── Dockerfile                 # API container build
├── mosquitto/config/mosquitto.conf
├── requirements.txt
├── xgb_final_model.pkl        # Trained XGBoost model
├── xgb_final_threshold.pkl    # Validated decision threshold (0.30)
├── feature_names.pkl          # Expected feature order for inference
└── code.ipynb                 # Training notebook: EDA, preprocessing, model selection, threshold validation
```

## Running it locally

**1. Start the MQTT broker**
```bash
docker-compose up mosquitto -d
```

**2. Start the API**
```bash
uvicorn main:app --reload
```

**3. Start a consumer** (either or both)
```bash
python subscriber.py          # terminal output
streamlit run dashboard.py    # live visual dashboard
```

**4. Start the simulated sensor stream**
```bash
python publisher.py --csv ai4i2020.csv --limit 300 --delay 0.5
```

The API is also available at `http://127.0.0.1:8000/docs` for interactive testing via Swagger UI.

## What this project demonstrates

- Handling severe class imbalance correctly, including recognizing and fixing threshold-selection data leakage
- Deploying a trained model behind a production-style REST API with input validation
- Understanding of IoT messaging patterns (MQTT pub/sub) as a decoupled, multi-consumer architecture
- Containerization of a multi-service system (broker + API) with Docker Compose
