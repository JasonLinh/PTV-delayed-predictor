# PTV METRO DELAY PREDICTOR
Real time data pipeline and machine learning model that predicts Melbourne metro train delays using live PTV GTFS-RT feed and weather data.

## OVERVIEW
This project collects live data on train delays from the PTV Open Data Api every 15 minutes, combining it with live Melbourne weather data from Open-Meteo storing it for later modelling. The goal of this project is to the duration of a train delay based on route, time of day, day of week, weather and service alert.

## PROJECT STRUCUTRE
```
ptv-delay-predictor/
├── collector.py     
├── requirements.txt
├── .env.example
└── data/
    ├── ptv_delays.csv
    └── collection_log.txt
```
## DATA COLLECTED
| Field | Description |
|---|---|
| `timestamp` | Snapshot time (Melbourne local) |
| `route_id` | Metro line identifier |
| `trip_id` | Individual service |
| `stop_id` | Station |
| `stop_sequence` | Stop position along the route |
| `delay_seconds` | Raw delay in seconds |
| `delay_minutes` | Delay in minutes |
| `is_delayed` | 1 if delay > 5 mins |
| `hour` | Hour of day (0-23) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `temperature` | °C |
| `windspeed` | km/h |
| `precipitation` | mm |
| `weather_code` | WMO weather condition code |
| `active_alerts` | Number of active network disruptions |
| `has_network_alert` | 1 if active |
## SET UP
- git clone https://github.com/JasonLinh/PTV-delayed-predictor
- cd PTV-delayed-predictor
- pip install -r requirements.txt
- cp .env.example .env         # then add your PTV_API_KEY
- python collector.py
## DEPLOY TO RAILWAY(RECOMMENDED)
- Push repo to GitHub
- Go to railway.app → New Project → Deploy from GitHub
- Add environment variable: PTV_API_KEY=your_key_here
- Add a Volume mounted at /app/data
- Deploy — collector runs automatically every 15 minutes 24/7
## MODELLING(in progress)
-Once enough data is collected, modelling will begin including
- Feature engineering
- Baseline testing: Linear regression
- Main model testing: Random Forest/ XGBoost
- Evaluation using RMSE, precision, recall and F1 score
- Explainability: Feature importance plots

## TECH STACK
- Data collection: Python, requests, gtfs-realtime-bindings, schedule
- Weather: Open-Meteo API (free, no key required)
- Storage: CSV on Railway persistent volume
- Deployment: Railway
- Modelling (upcoming): pandas, scikit-learn, XGBoost, SHAP, Matplotlib
## RATIONALE
Melbourne's metropolitan network is used by millions of passengers per year. Even small improvements in delay predictions can help commuters better plan their journeys and help operators allocate resources more effectively.
