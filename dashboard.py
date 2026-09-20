"""
Live predictive-maintenance dashboard.

Subscribes to the same MQTT topic as subscriber.py, calls the FastAPI
/predict endpoint for each incoming reading, and displays a live table +
risk chart in Streamlit.

This does NOT replace subscriber.py — it's an alternative "frontend" for
the same stream, useful for demos. You can run this instead of (or
alongside) subscriber.py.

Run (with mosquitto + main.py + publisher.py already running):
    streamlit run dashboard.py
"""

import json
import queue
import threading
import time
from collections import deque
from datetime import datetime

import pandas as pd
import paho.mqtt.client as mqtt
import requests
import streamlit as st

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "factory/machine1/sensors"
API_URL = "http://localhost:8000/predict"
MAX_ROWS = 50  # how many recent readings to keep on screen

st.set_page_config(page_title="Predictive Maintenance — Live Feed", layout="wide")


# ---------------------------------------------------------------------------
# Background MQTT listener
#
# Streamlit reruns the whole script top-to-bottom on every interaction, so
# the MQTT client can't live in normal script state — it has to run in a
# background thread that persists across reruns, pushing results into a
# thread-safe queue that the main script drains each rerun.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_message_queue():
    return queue.Queue()


def mqtt_worker(msg_queue: queue.Queue):
    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        true_label = payload.pop("_true_label", None)
        udi = payload.pop("_udi", None)

        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            msg_queue.put({"error": str(e), "udi": udi})
            return

        msg_queue.put(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "udi": udi,
                "type": payload["type"],
                "air_temp_k": payload["air_temperature_k"],
                "process_temp_k": payload["process_temperature_k"],
                "rpm": payload["rotational_speed_rpm"],
                "torque_nm": payload["torque_nm"],
                "tool_wear_min": payload["tool_wear_min"],
                "failure_probability": result["failure_probability"],
                "prediction": result["prediction"],
                "true_label": true_label,
            }
        )

    def on_connect(client, userdata, flags, rc):
        client.subscribe(TOPIC)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_forever()


@st.cache_resource
def start_background_listener():
    msg_queue = get_message_queue()
    thread = threading.Thread(target=mqtt_worker, args=(msg_queue,), daemon=True)
    thread.start()
    return thread


start_background_listener()

# ---------------------------------------------------------------------------
# Session state for accumulated readings (per browser session)
# ---------------------------------------------------------------------------
if "readings" not in st.session_state:
    st.session_state.readings = deque(maxlen=MAX_ROWS)

msg_queue = get_message_queue()
drained = 0
while not msg_queue.empty():
    item = msg_queue.get()
    if "error" not in item:
        st.session_state.readings.appendleft(item)
    drained += 1

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔧 Predictive Maintenance — Live Sensor Feed")
st.caption(
    f"Subscribed to MQTT topic `{TOPIC}` — readings are scored live via the "
    f"FastAPI `/predict` endpoint. Auto-refreshes every few seconds."
)

readings = list(st.session_state.readings)

if not readings:
    st.info("Waiting for sensor readings... make sure publisher.py is running.")
else:
    df = pd.DataFrame(readings)

    col1, col2, col3 = st.columns(3)
    col1.metric("Readings received", len(readings))
    col2.metric("Failures predicted", int(df["prediction"].sum()))
    latest_risk = df.iloc[0]["failure_probability"]
    col3.metric("Latest failure risk", f"{latest_risk:.1%}")

    st.subheader("Failure risk over time")
    chart_df = df.iloc[::-1].set_index("time")[["failure_probability"]]
    st.line_chart(chart_df)

    st.subheader("Recent readings")

    def highlight_failures(row):
        color = "background-color: #ffcccc" if row["prediction"] == 1 else ""
        return [color] * len(row)

    st.dataframe(
        df.style.apply(highlight_failures, axis=1).format({"failure_probability": "{:.4f}"}),
        use_container_width=True,
    )

# Poll for new messages every 2 seconds by rerunning the script.
# (Simpler and more reliable for a local demo than a full page meta-refresh,
# which can drop session state on reload.)
time.sleep(2)
st.rerun()