
# PBL ML Frontend + Flask Backend

## Files
- app.py — Flask server that trains a RandomForest on startup and serves `/predict`.
- templates/index.html — Simple UI that posts to `/predict`.
- requirements.txt — Python deps.

## Setup (Local)
1) Put your dataset CSV next to `app.py` with the exact name:
   `vehicles_with_pollution_with_weather_with_area.csv`

2) Create and activate a virtual environment (optional but recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3) Install deps
   ```bash
   pip install -r requirements.txt
   ```

4) Run the server
   ```bash
   python app.py
   ```

5) Open http://localhost:5000 in your browser.

## Predict via cURL (optional)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"co2":120,"nox":80,"pm25":60}'
```
