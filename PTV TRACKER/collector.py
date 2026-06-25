from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
import requests
import pandas as pd
import schedule
from datetime import datetime
from google.transit import gtfs_realtime_pb2
from flask import Flask

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get('PTV_API_KEY')

FEEDS = {
    'metro_train': (
        'https://api.opendata.transport.vic.gov.au'
        '/opendata/public-transport/gtfs/realtime/v1/metro/trip-updates'
    ),
    'metro_alerts': (
        'https://api.opendata.transport.vic.gov.au'
        '/opendata/public-transport/gtfs/realtime/v1/metro/service-alerts'
    ),
}

DATA_FILE = 'data/ptv_delays.csv'
LOG_FILE  = 'data/collection_log.txt'

# ── Flask keep-alive (required for Railway) ───────────────────────────────────
app = Flask(__name__)

@app.route('/')
def home():
    return "PTV collector running ✅"

def run_web():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

# ── Data collection ───────────────────────────────────────────────────────────
def get_delays(feed_url: str, snapshot_time: datetime) -> pd.DataFrame:
    """Fetch GTFS-RT trip updates and return a flat DataFrame."""
    response = requests.get(
        feed_url,
        headers={'Ocp-Apim-Subscription-Key': API_KEY},
        timeout=15,
    )
    response.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)

    records = []
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue

        trip      = entity.trip_update
        route_id  = trip.trip.route_id
        trip_id   = trip.trip.trip_id

        for stop_update in trip.stop_time_update:
            delay_secs = None

            if stop_update.HasField('departure'):
                delay_secs = stop_update.departure.delay
            elif stop_update.HasField('arrival'):
                delay_secs = stop_update.arrival.delay

            records.append({
                'timestamp':      snapshot_time,
                'route_id':       route_id,
                'trip_id':        trip_id,
                'stop_id':        stop_update.stop_id,
                'stop_sequence':  stop_update.stop_sequence,
                'delay_seconds':  delay_secs,
                'delay_minutes':  round(delay_secs / 60, 2) if delay_secs else 0,
                'is_delayed':     1 if delay_secs and delay_secs > 300 else 0,
                'hour':           snapshot_time.hour,
                'day_of_week':    snapshot_time.weekday(),
            })

    return pd.DataFrame(records)


def get_alerts() -> dict:
    response = requests.get(
        FEEDS['metro_alerts'],
        headers={
            'Ocp-Apim-Subscription-Key': API_KEY,
            'KeyID': API_KEY
        },
        timeout=15,
    )
    response.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(response.content)

    effects = []
    causes  = []
    for entity in feed.entity:
        if entity.HasField('alert'):
            effects.append(entity.alert.effect)
            causes.append(entity.alert.cause)

    return {
        'active_alerts':     len(effects),
        'has_network_alert': 1 if len(effects) > 0 else 0,
        'alert_effects':     ','.join(str(e) for e in effects) if effects else '',
        'alert_causes':      ','.join(str(c) for c in causes) if causes else '',
    }


def collect_snapshot():
    """Single collection run: fetch delays, alerts, weather and append to CSV."""
    os.makedirs('data', exist_ok=True)
    now = datetime.now()
    print(f"[{now}] Collecting snapshot...")

    try:
        weather = get_weather()
        alerts  = get_alerts()
        df      = get_delays(FEEDS['metro_train'], snapshot_time=now)

        if df.empty:
            print("  No records returned — skipping.")
            return

        for key, val in weather.items():
            df[key] = val

        for key, val in alerts.items():
            df[key] = val

        file_exists = os.path.isfile(DATA_FILE)
        df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)

        with open(LOG_FILE, 'a') as f:
            f.write(f"{now} — {len(df)} records saved | alerts: {alerts['active_alerts']}\n")

        print(f"  ✅ Saved {len(df)} records | delayed: {df['is_delayed'].sum()} | alerts: {alerts['active_alerts']}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        with open(LOG_FILE, 'a') as f:
            f.write(f"{now} — ERROR: {e}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=run_web, daemon=True).start()

    collect_snapshot()

    schedule.every(5).minutes.do(collect_snapshot)
    print("Scheduler running — collecting every 5 minutes.")

    while True:
        schedule.run_pending()
        time.sleep(1)