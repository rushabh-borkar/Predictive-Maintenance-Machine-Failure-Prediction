"""
MQTT subscriber / inference bridge.

Listens for simulated sensor readings on the MQTT broker and forwards
each one to the FastAPI /predict endpoint, printing the model's live
failure prediction as if this were a real-time monitoring dashboard.

Run (after publisher.py is running):
    python subscriber.py
"""

import json

import paho.mqtt.client as mqtt
import requests

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "factory/machine1/sensors"
API_URL = "http://localhost:8000/predict"


def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker (rc={rc}), subscribing to '{TOPIC}'")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    true_label = payload.pop("_true_label", None)
    udi = payload.pop("_udi", None)

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        print(f"[UDI {udi}] Prediction request failed: {e}")
        return

    flag = "⚠️  FAILURE PREDICTED" if result["prediction"] == 1 else "OK"
    match = ""
    if true_label is not None:
        correct = int(result["prediction"] == true_label)
        match = f" | true_label={true_label} | {'✓ correct' if correct else '✗ WRONG'}"

    print(
        f"[UDI {udi}] prob={result['failure_probability']:.4f} "
        f"threshold={result['threshold_used']} -> {flag}{match}"
    )


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    print("Listening for sensor readings... (Ctrl+C to stop)")
    client.loop_forever()


if __name__ == "__main__":
    main()