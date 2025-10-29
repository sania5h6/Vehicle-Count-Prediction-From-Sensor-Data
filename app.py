
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

# --------- Configuration ----------
DATA_FILENAME = "vehicles_with_pollution_with_weather_with_area.csv"  # Put this file in the project root
FEATURES = ['CO2 Emissions (g/km)', 'NOx Emissions (mg/km)', 'PM2.5 Concentration (µg/m³)']

# --------- Training on startup (simple) ----------
def load_and_train():
    # Try to read dataset from local path
    df = pd.read_csv(DATA_FILENAME)

    # Recreate your Vehicle Count logic (defensive coding)
    rng = np.random.RandomState(42)
    vehicle_count = (df['CO2 Emissions (g/km)'] * 0.5 +
                     df['NOx Emissions (mg/km)'] * 0.3 +
                     rng.normal(0, 20, len(df)))
    vehicle_count = pd.Series(vehicle_count).fillna(0).astype(int).clip(lower=0)
    df['Vehicle Count'] = vehicle_count

    # Features/target
    X = df[FEATURES].copy()
    y = df['Vehicle Count'].copy()

    # Basic split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Train RF (sane defaults for demo)
    model = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train_imp, y_train)

    # Quick metrics
    y_pred = model.predict(X_test_imp)
    metrics = dict(
        r2=float(r2_score(y_test, y_pred)),
        mae=float(mean_absolute_error(y_test, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred)))
    )
    return model, imputer, metrics

model, imputer, metrics = load_and_train()

# --------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html", metrics=metrics, features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        co2 = float(data.get("co2"))
        nox = float(data.get("nox"))
        pm25 = float(data.get("pm25"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid inputs. Provide numeric co2, nox, pm25."}), 400

    X_new = np.array([[co2, nox, pm25]], dtype=float)
    X_new_imp = imputer.transform(X_new)
    y_hat = float(model.predict(X_new_imp)[0])

    # Simple bucketing for a friendly label (purely illustrative)
    if y_hat < 50:
        label = "Low traffic expected"
    elif y_hat < 150:
        label = "Moderate traffic expected"
    else:
        label = "High traffic expected"

    return jsonify({
        "ok": True,
        "prediction": int(round(y_hat)),
        "label": label
    })

if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
