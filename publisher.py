"""
Simulated IoT sensor publisher.

Reads rows from the AI4I dataset (ideally your held-out test set, so you
can compare predictions against known labels) and publishes each one as
an MQTT message, as if a real machine's sensors were streaming readings.

Run:
    python publisher.py --csv ai4i2020.csv --delay 2
"""

import argparse
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "factory/machine1/sensors"


def row_to_payload(row: pd.Series) -> dict:
    """Map a raw AI4I row to the JSON shape the FastAPI /predict endpoint expects."""
    return {
        "type": row["Type"],
        "air_temperature_k": float(row["Air temperature [K]"]),
        "process_temperature_k": float(row["Process temperature [K]"]),
        "rotational_speed_rpm": float(row["Rotational speed [rpm]"]),
        "torque_nm": float(row["Torque [Nm]"]),
        "tool_wear_min": float(row["Tool wear [min]"]),
        # kept only for your own comparison in subscriber logs — not sent
        # to the model, since the model never sees the true label
        "_true_label": int(row["Machine failure"]) if "Machine failure" in row else None,
        "_udi": int(row["UDI"]) if "UDI" in row else None,
    }


def main(csv_path: str, delay: float, limit: int | None):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    client = mqtt.Client()
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    print(f"Publishing {len(df)} readings to '{TOPIC}' every {delay}s...")

    try:
        for _, row in df.iterrows():
            payload = row_to_payload(row)
            client.publish(TOPIC, json.dumps(payload))
            print(f"Published UDI {payload['_udi']} (true_label={payload['_true_label']})")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to AI4I CSV (ideally your test split)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between readings")
    parser.add_argument("--limit", type=int, default=None, help="Only publish first N rows")
    args = parser.parse_args()

    main(args.csv, args.delay, args.limit)