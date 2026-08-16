import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
import joblib
import os
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =========================================================
# SMART TRAFFIC AI
# AI-Powered Traffic Monitoring & Prediction System
# =========================================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM DESIGN
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #102a43, #176b87);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    border: 1px solid #e8edf3;
}

.metric-title {
    color: #64748b;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #102a43;
}

.status {
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
}
/* =========================================================
   TRAFFIC ANALYTICS
   ========================================================= */

.analytics-card {
    padding: 10px 5px 5px 5px;
}

.analytics-title {
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
}

.analytics-subtitle {
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
    margin-bottom: 10px;
}

.analytics-image-title {
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}

.analytics-image-subtitle {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 10px;
}
/* =========================================================
   SMART TRAFFIC HERO
   ========================================================= */

.hero-banner {
    min-height: 330px;
    border-radius: 28px;
    background-size: cover;
    background-position: center;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    margin-bottom: 30px;
    box-shadow:
        0 20px 50px rgba(15, 23, 42, 0.25);
}

.hero-banner::before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    right: -80px;
    top: -80px;
    background: rgba(56, 189, 248, 0.18);
    filter: blur(5px);
}

.hero-content {
    position: relative;
    z-index: 2;
    padding: 42px;
    max-width: 750px;
}

.hero-content h1 {
    color: white !important;
    font-size: 48px !important;
    font-weight: 900 !important;
    letter-spacing: 1px;
    margin: 14px 0 8px 0;
}

.hero-content p {
    color: #dbeafe;
    font-size: 19px;
    margin-bottom: 22px;
}

.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(74, 222, 128, 0.45);
    color: #bbf7d0;
    padding: 7px 14px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.live-dot {
    width: 9px;
    height: 9px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 12px #22c55e;
}

.hero-tags {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.hero-tags span {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: white;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    backdrop-filter: blur(8px);
}
/* =========================================================
   COLORFUL DASHBOARD CARDS
   ========================================================= */

.metric-card {
    min-height: 175px;
    padding: 22px;
    border-radius: 20px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15);
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 38px rgba(15, 23, 42, 0.22);
}

.metric-card::after {
    content: "";
    position: absolute;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    right: -35px;
    bottom: -45px;
    background: rgba(255, 255, 255, 0.14);
}

.blue-card {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
}

.purple-card {
    background: linear-gradient(135deg, #7c3aed, #c026d3);
}

.orange-card {
    background: linear-gradient(135deg, #ea580c, #f59e0b);
}

.green-card {
    background: linear-gradient(135deg, #059669, #22c55e);
}

.metric-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

.metric-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    opacity: 0.9;
}

.metric-number {
    font-size: 27px;
    font-weight: 900;
    margin-top: 5px;
}

.metric-subtitle {
    font-size: 12px;
    margin-top: 6px;
    opacity: 0.85;
}
/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 18px;
    margin-bottom: 20px;
    padding: 16px 20px;
    border-radius: 17px;
    background: linear-gradient(
        135deg,
        rgba(37, 99, 235, 0.10),
        rgba(124, 58, 237, 0.08)
    );
    border-left: 5px solid #2563eb;
}

.section-icon {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    font-size: 25px;
    box-shadow: 0 7px 18px rgba(37, 99, 235, 0.25);
}

.section-title {
    font-size: 20px;
    font-weight: 850;
    color: #0f172a;
    letter-spacing: 0.5px;
}

.section-description {
    color: #64748b;
    font-size: 13px;
    margin-top: 3px;
}
/* =========================================================
   AI PREDICTION RESULT
   ========================================================= */

.prediction-card {
    min-height: 245px;
    padding: 25px;
    border-radius: 22px;
    background: linear-gradient(145deg, #0f172a, #172554);
    color: white;
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.20);
    position: relative;
    overflow: hidden;
}

.prediction-card::after {
    content: "🤖";
    position: absolute;
    right: 20px;
    bottom: -25px;
    font-size: 120px;
    opacity: 0.08;
}

.prediction-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
}

.prediction-live {
    color: #4ade80;
    font-size: 11px;
}

.prediction-number {
    font-size: 48px;
    font-weight: 900;
    margin-top: 25px;
}

.prediction-unit {
    color: #94a3b8;
    font-size: 14px;
}

.traffic-status {
    margin-top: 22px;
    padding: 11px 15px;
    border-radius: 12px;
    display: inline-block;
    font-size: 16px;
    font-weight: 850;
}

.traffic-status.low {
    background: rgba(34, 197, 94, 0.18);
    color: #86efac;
}

.traffic-status.moderate {
    background: rgba(234, 179, 8, 0.18);
    color: #fde047;
}

.traffic-status.high {
    background: rgba(249, 115, 22, 0.18);
    color: #fdba74;
}

.traffic-status.critical {
    background: rgba(239, 68, 68, 0.18);
    color: #fca5a5;
}


/* =========================================================
   AI RECOMMENDATION
   ========================================================= */

.recommendation-card {
    min-height: 245px;
    padding: 25px;
    border-radius: 22px;
    background: linear-gradient(145deg, #ffffff, #f5f3ff);
    border: 1px solid #ddd6fe;
    box-shadow: 0 15px 35px rgba(76, 29, 149, 0.10);
}

.recommendation-title {
    font-size: 17px;
    font-weight: 850;
    color: #5b21b6;
}

.recommendation-message {
    margin-top: 18px;
    font-size: 17px;
    line-height: 1.6;
    color: #334155;
}

.recommendation-details {
    display: flex;
    gap: 15px;
    margin-top: 25px;
}

.recommendation-details div {
    flex: 1;
    padding: 12px;
    border-radius: 12px;
    background: #ede9fe;
}

.recommendation-details span {
    display: block;
    font-size: 11px;
    color: #64748b;
    margin-bottom: 5px;
}

.recommendation-details strong {
    color: #312e81;
    font-size: 14px;
}
.analytics-image {
    width: 100%;
    height: 230px;
    object-fit: cover;
    border-radius: 20px;
    margin-top: 15px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15);
}
.analytics-card {
    padding: 15px 18px 5px 18px;
    margin-bottom: 5px;
}

.analytics-title {
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
}

.analytics-subtitle {
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
}

.analytics-image-card {
    height: 300px;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.15);
    margin-top: 15px;
}

.analytics-image-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.peak-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
    min-height: 150px;
}

.peak-card:hover {
    transform: translateY(-4px);
}

.peak-icon {
    font-size: 26px;
    margin-bottom: 10px;
}

.peak-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #64748b;
}

.peak-value {
    font-size: 28px;
    font-weight: 900;
    color: #0f172a;
    margin-top: 6px;
}

.peak-description {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 6px;
}

.peak-card.critical {
    border: 1px solid #fecaca;
    background: linear-gradient(145deg, #ffffff, #fff1f2);
}

.peak-card.high {
    border: 1px solid #fed7aa;
    background: linear-gradient(145deg, #ffffff, #fff7ed);
}

.peak-card.normal {
    border: 1px solid #bbf7d0;
    background: linear-gradient(145deg, #ffffff, #f0fdf4);
}
.action-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
    min-height: 190px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.action-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
}

.action-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

.action-title {
    font-size: 14px;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: 0.5px;
}

.action-text {
    font-size: 13px;
    line-height: 1.6;
    color: #64748b;
    margin-top: 10px;
}

.action-status {
    display: inline-block;
    margin-top: 15px;
    padding: 6px 10px;
    border-radius: 8px;
    background: #eff6ff;
    color: #2563eb;
    font-size: 10px;
    font-weight: 800;
}
/* =========================================================
   SYSTEM STATUS
   ========================================================= */

.system-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.07);
    text-align: center;
    min-height: 125px;
    transition: transform 0.2s ease;
}

.system-card:hover {
    transform: translateY(-4px);
}

.system-icon {
    font-size: 26px;
    margin-bottom: 8px;
}

.system-name {
    font-size: 11px;
    font-weight: 800;
    color: #334155;
    letter-spacing: 0.5px;
}

.system-status {
    margin-top: 8px;
    font-size: 11px;
    font-weight: 800;
    color: #16a34a;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

MODEL_PATH = "model/traffic_model.pkl"
FEATURE_PATH = "model/features.pkl"
DATA_PATH = "data/Metro_Interstate_Traffic_Volume.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
# ---------------------------------------------------------
# VEHICLE DETECTION MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_vehicle_model():
    return YOLO("yolo11n.pt")


vehicle_model = load_vehicle_model()


try:
    model, features = load_model()
    data = load_data()
except Exception as e:
    st.error("Unable to load the AI model or dataset.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# HEADER / HERO SECTION
# ---------------------------------------------------------

import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


hero_image = get_image_base64("assets/traffic_hero.jpeg")

st.markdown(
    f"""
<div class="hero-banner" style="
    background-image:
    linear-gradient(
        90deg,
        rgba(7, 15, 35, 0.96) 0%,
        rgba(15, 23, 42, 0.82) 45%,
        rgba(15, 23, 42, 0.35) 100%
    ),
    url('data:image/jpeg;base64,{hero_image}');
">

<div class="hero-content">

<div class="live-badge">
<span class="live-dot"></span>
AI SYSTEM ONLINE
</div>

<h1>🚦 SMART TRAFFIC AI</h1>

<p>
Intelligent Traffic Monitoring,
Prediction & Congestion Intelligence
</p>

<div class="hero-tags">
<span>🤖 AI Powered</span>
<span>📊 Real-Time Insights</span>
<span>🛣️ Smart Mobility</span>
</div>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🚦 Traffic Control")

st.sidebar.markdown("---")

st.sidebar.subheader("Prediction Inputs")

hour = st.sidebar.slider(
    "🕐 Hour of Day",
    min_value=0,
    max_value=23,
    value=8
)

temperature = st.sidebar.number_input(
    "🌡️ Temperature",
    value=288.0,
    step=0.5
)

rain = st.sidebar.number_input(
    "🌧️ Rain (1h)",
    min_value=0.0,
    value=0.0,
    step=0.1
)

snow = st.sidebar.number_input(
    "❄️ Snow (1h)",
    min_value=0.0,
    value=0.0,
    step=0.1
)

clouds = st.sidebar.slider(
    "☁️ Cloud Coverage",
    min_value=0,
    max_value=100,
    value=40
)

weather = st.sidebar.selectbox(
    "🌦️ Weather",
    [
        "Clear",
        "Clouds",
        "Rain",
        "Snow",
        "Mist",
        "Fog",
        "Drizzle",
        "Thunderstorm"
    ]
)

predict_button = st.sidebar.button(
    "🚀 ANALYZE TRAFFIC",
    use_container_width=True
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "🔄 RESET CONTROLS",
    use_container_width=True
):
    st.rerun()


# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card blue-card">
        <div class="metric-icon">📊</div>
        <div class="metric-label">DATASET RECORDS</div>
        <div class="metric-number">{len(data):,}</div>
        <div class="metric-subtitle">Traffic observations</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card purple-card">
        <div class="metric-icon">🚗</div>
        <div class="metric-label">AVERAGE TRAFFIC</div>
        <div class="metric-number">{data['traffic_volume'].mean():,.0f}</div>
        <div class="metric-subtitle">Average vehicles</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card orange-card">
        <div class="metric-icon">📈</div>
        <div class="metric-label">MAXIMUM TRAFFIC</div>
        <div class="metric-number">{data['traffic_volume'].max():,.0f}</div>
        <div class="metric-subtitle">Highest recorded volume</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card green-card">
        <div class="metric-icon">🤖</div>
        <div class="metric-label">AI SYSTEM</div>
        <div class="metric-number">ONLINE</div>
        <div class="metric-subtitle">Prediction model active</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ---------------------------------------------------------
# TRAFFIC PREDICTION
# ---------------------------------------------------------

st.markdown("""
<div class="section-header">
    <div class="section-icon">🤖</div>
    <div>
        <div class="section-title">AI TRAFFIC PREDICTION</div>
        <div class="section-description">
            Predict traffic volume and identify congestion levels using AI
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame({
            "holiday": ["None"],
            "temp": [temperature],
            "rain_1h": [rain],
            "snow_1h": [snow],
            "clouds_all": [clouds],
            "weather_main": [weather],
            "weather_description": ["sky is clear"],
            "hour": [hour],
            "day": [1],
            "month": [1],
            "weekday": [1]
        })


        # -------------------------------------------------
        # CONVERT CATEGORICAL COLUMNS
        # -------------------------------------------------

        input_encoded = pd.get_dummies(
            input_data,
            columns=[
                "holiday",
                "weather_main",
                "weather_description"
            ]
        )


        # -------------------------------------------------
        # MATCH TRAINING FEATURES
        # -------------------------------------------------

        input_encoded = input_encoded.reindex(
            columns=features,
            fill_value=0
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_encoded)[0]

        prediction = max(0, prediction)


                # -------------------------------------------------
        # CONGESTION LEVEL
        # -------------------------------------------------

        if prediction < 3000:

            level = "LOW"
            emoji = "🟢"
            message = "Traffic conditions are normal."

        elif prediction < 6000:

            level = "MODERATE"
            emoji = "🟡"
            message = "Traffic is building up. Monitor the corridor."

        elif prediction < 8000:

            level = "HIGH"
            emoji = "🟠"
            message = "Heavy traffic detected. Traffic management action recommended."

        else:

            level = "CRITICAL"
            emoji = "🔴"
            message = "Severe congestion predicted. Immediate intervention recommended."


        # -------------------------------------------------
        # SAVE PREDICTION TO HISTORY
        # -------------------------------------------------

        st.session_state.prediction_history.append({
            "Time": f"{hour:02d}:00",
            "Predicted Vehicles": round(prediction),
            "Traffic Level": f"{emoji} {level}"
        })

                # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "📊 Predicted Traffic Volume",
                f"{prediction:,.0f} vehicles",
                f"{emoji} {level}"
            )


        with r2:

            st.markdown("### 🧠 AI Recommendation")

            st.write(message)

            st.write(f"🕐 **Prediction time:** {hour:02d}:00")

            st.write(f"🌤️ **Weather:** {weather}")


               # -------------------------------------------------
        # UNDERSTAND PREDICTION
        # -------------------------------------------------

        st.write("")

        with st.expander("🔍 Understand This Prediction"):

            st.markdown(f"""
            ### 🚦 Traffic Interpretation

            **Predicted Volume:** {prediction:,.0f} vehicles

            **Congestion Level:** {emoji} {level}

            **Prediction Time:** {hour:02d}:00

            **Weather Condition:** {weather}

            ### 🧠 What the AI is telling you

            {message}

            The prediction is generated using the traffic and
            weather conditions selected in the Traffic Control panel.
            """)


    except Exception as e:

        st.error("Prediction could not be generated.")
        st.exception(e)
# ---------------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------------

st.write("")

st.markdown("""
<div class="section-header">
    <div class="section-icon">📜</div>
    <div>
        <div class="section-title">PREDICTION HISTORY</div>
        <div class="section-description">
            Review traffic predictions generated during this session
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    # -------------------------------------------------
    # HISTORY SUMMARY
    # -------------------------------------------------

    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric(
            "📊 Total Predictions",
            len(history_df)
        )

    with h2:
        st.metric(
            "🚗 Highest Traffic",
            f"{history_df['Predicted Vehicles'].max():,.0f}"
        )

    with h3:
        st.metric(
            "📉 Lowest Traffic",
            f"{history_df['Predicted Vehicles'].min():,.0f}"
        )

    st.write("")

    # -------------------------------------------------
    # HISTORY TABLE
    # -------------------------------------------------

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:

    st.info(
        "📊 No predictions recorded yet. "
        "Run a traffic analysis to create prediction history."
    )
    # ---------------------------------------------------------
# LIVE CAMERA MONITORING
# ---------------------------------------------------------

st.write("")

st.markdown("""
<div class="section-header">
    <div class="section-icon">📷</div>
    <div>
        <div class="section-title">LIVE CAMERA MONITORING</div>
        <div class="section-description">
            Capture a traffic scene for visual monitoring
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

camera_image = st.camera_input(
    "📷 Capture Traffic Scene"
)

if camera_image is not None:

    # Read captured image
    image_bytes = camera_image.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    # -------------------------------------------------
    # AI VEHICLE DETECTION
    # -------------------------------------------------

    results = vehicle_model(
        image,
        conf=0.40,
        iou=0.50,
        imgsz=640
    )

    vehicle_classes = [2, 3, 5, 7]

    vehicle_count = 0

    for result in results:

        if result.boxes is None:
            continue

        for cls, conf in zip(
            result.boxes.cls,
            result.boxes.conf
        ):

            class_id = int(cls)
            confidence = float(conf)

            if class_id in vehicle_classes and confidence >= 0.40:
                vehicle_count += 1

    # Draw detection boxes
    annotated_image = results[0].plot()

    st.image(
        annotated_image,
        channels="BGR",
        caption="🤖 AI Vehicle Detection",
        use_container_width=True
    )

    # -------------------------------------------------
    # CAMERA TRAFFIC LEVEL
    # -------------------------------------------------

    if vehicle_count >= 15:

        camera_level = "🔴 HIGH"

        camera_message = (
            "High vehicle density detected. "
            "Traffic authorities should monitor this corridor."
        )

    elif vehicle_count >= 7:

        camera_level = "🟡 MODERATE"

        camera_message = (
            "Moderate vehicle density detected. "
            "Continue monitoring the corridor."
        )

    else:

        camera_level = "🟢 LOW"

        camera_message = (
            "Low vehicle density detected. "
            "Traffic conditions appear normal."
        )

    # -------------------------------------------------
    # CAMERA RESULTS
    # -------------------------------------------------

    st.markdown("### 🚦 Camera Traffic Analysis")

    cam_col1, cam_col2 = st.columns(2)

    with cam_col1:

        st.metric(
            "🚗 Vehicles Detected",
            vehicle_count
        )

    with cam_col2:

        st.metric(
            "🚦 Camera Traffic Level",
            camera_level
        )

    st.info(camera_message)


# ---------------------------------------------------------
# TRAFFIC ANALYTICS
# ---------------------------------------------------------

st.write("")
st.markdown("""
<div class="section-header">
    <div class="section-icon">📊</div>
    <div>
        <div class="section-title">TRAFFIC ANALYTICS</div>
        <div class="section-description">
            Explore traffic patterns and identify peak congestion periods
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Convert date
data["date_time"] = pd.to_datetime(
    data["date_time"],
    errors="coerce"
)

data["hour"] = data["date_time"].dt.hour


# Hourly traffic
hourly = (
    data.groupby("hour")["traffic_volume"]
    .mean()
    .reset_index()
)

col_chart, col_image = st.columns([1.7, 1])

with col_chart:

    st.markdown(
        "### 📈 Average Traffic by Hour"
    )

    st.caption(
        "Traffic volume across different hours of the day"
    )

    st.line_chart(
        hourly.set_index("hour"),
        use_container_width=True
    )


with col_image:

    st.markdown("### 🚦 Traffic Intelligence")

    st.caption(
        "Visual overview of urban traffic conditions"
    )

    st.image(
        "assets/traffic_analysis.jpeg",
        use_container_width=True
    )


# ---------------------------------------------------------
# PEAK HOURS
# ---------------------------------------------------------

st.markdown("""
<div class="section-header">
    <div class="section-icon">⏰</div>
    <div>
        <div class="section-title">PEAK TRAFFIC ANALYSIS</div>
        <div class="section-description">
            Identify the busiest period and understand congestion severity
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


peak_hour = hourly.loc[
    hourly["traffic_volume"].idxmax()
]

peak_hour_value = int(peak_hour["hour"])
peak_volume = peak_hour["traffic_volume"]


if peak_volume > 7000:
    status = "🔴 CRITICAL"
    status_class = "critical"

elif peak_volume > 5000:
    status = "🟠 HIGH"
    status_class = "high"

else:
    status = "🟢 NORMAL"
    status_class = "normal"


peak_col1, peak_col2, peak_col3 = st.columns(3)


with peak_col1:
    st.markdown(
        f"""
        <div class="peak-card">
            <div class="peak-icon">🕐</div>
            <div class="peak-label">PEAK HOUR</div>
            <div class="peak-value">{peak_hour_value:02d}:00</div>
            <div class="peak-description">
                Busiest traffic period
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with peak_col2:
    st.markdown(
        f"""
        <div class="peak-card">
            <div class="peak-icon">🚗</div>
            <div class="peak-label">PEAK VOLUME</div>
            <div class="peak-value">{peak_volume:,.0f}</div>
            <div class="peak-description">
                Average vehicles recorded
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with peak_col3:
    st.markdown(
        f"""
        <div class="peak-card {status_class}">
            <div class="peak-icon">🚦</div>
            <div class="peak-label">PEAK STATUS</div>
            <div class="peak-value">{status}</div>
            <div class="peak-description">
                Congestion severity
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# GOVERNMENT ACTION PANEL
# ---------------------------------------------------------

st.markdown("""
<div class="section-header">
    <div class="section-icon">🏛️</div>
    <div>
        <div class="section-title">TRAFFIC AUTHORITY ACTION CENTER</div>
        <div class="section-description">
            AI-assisted recommendations for smarter traffic management
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


action_col1, action_col2, action_col3 = st.columns(3)


with action_col1:
    st.markdown("""
    <div class="action-card">
        <div class="action-icon">🚦</div>
        <div class="action-title">SIGNAL OPTIMIZATION</div>
        <div class="action-text">
            Dynamically adjust traffic signal timing
            during predicted peak congestion periods.
        </div>
        <div class="action-status">
            ⚡ AI RECOMMENDED
        </div>
    </div>
    """, unsafe_allow_html=True)


with action_col2:
    st.markdown("""
    <div class="action-card">
        <div class="action-icon">📢</div>
        <div class="action-title">CONGESTION ALERT</div>
        <div class="action-text">
            Issue early warnings to commuters when
            high traffic conditions are predicted.
        </div>
        <div class="action-status">
            📡 MONITORING
        </div>
    </div>
    """, unsafe_allow_html=True)


with action_col3:
    st.markdown("""
    <div class="action-card">
        <div class="action-icon">🛣️</div>
        <div class="action-title">ROUTE MANAGEMENT</div>
        <div class="action-text">
            Recommend alternate routes and deploy
            traffic personnel toward congested corridors.
        </div>
        <div class="action-status">
            🤖 SMART RESPONSE
        </div>
    </div>
    """, unsafe_allow_html=True)
# ---------------------------------------------------------
# SYSTEM STATUS
# ---------------------------------------------------------

st.write("")

st.markdown("""
<div class="section-header">
    <div class="section-icon">🛰️</div>
    <div>
        <div class="section-title">SYSTEM STATUS</div>
        <div class="section-description">
            Smart Traffic AI platform health and component status
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


status_col1, status_col2, status_col3, status_col4 = st.columns(4)


with status_col1:
    st.markdown("""
    <div class="system-card">
        <div class="system-icon">🤖</div>
        <div class="system-name">AI SYSTEM</div>
        <div class="system-status">● ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View AI System", key="ai_system"):
        st.info(
            "🤖 AI System: The Smart Traffic AI application is "
            "running and ready to analyze traffic conditions."
        )


with status_col2:
    st.markdown("""
    <div class="system-card">
        <div class="system-icon">📡</div>
        <div class="system-name">PREDICTION ENGINE</div>
        <div class="system-status">● ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Prediction Engine", key="prediction_engine"):
        st.info(
            "📡 Prediction Engine: The trained machine-learning model "
            "uses the selected traffic and weather inputs to estimate "
            "traffic volume."
        )


with status_col3:
    st.markdown("""
    <div class="system-card">
        <div class="system-icon">📊</div>
        <div class="system-name">DATA ANALYTICS</div>
        <div class="system-status">● CONNECTED</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Data Analytics", key="data_analytics"):
        st.info(
            "📊 Data Analytics: Historical traffic data is analyzed "
            "to identify hourly traffic patterns and peak congestion periods."
        )


with status_col4:
    st.markdown("""
    <div class="system-card">
        <div class="system-icon">🧠</div>
        <div class="system-name">MODEL STATUS</div>
        <div class="system-status">● READY</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Model Status", key="model_status"):
        st.info(
            "🧠 Model Status: The trained traffic prediction model "
            "has been loaded successfully and is ready to generate predictions."
        )
        # ---------------------------------------------------------
# AI DECISION SUMMARY
# ---------------------------------------------------------

st.write("")

st.markdown("""
<div class="section-header">
    <div class="section-icon">🧠</div>
    <div>
        <div class="section-title">AI DECISION SUMMARY</div>
        <div class="section-description">
            Quick interpretation of the traffic intelligence
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.markdown("### 🚦 Current Traffic Guidance")

    st.info(
        "Use the Traffic Control panel to select different "
        "conditions and run a new AI traffic prediction."
    )


with summary_col2:

    st.markdown("### 📊 Planning Insight")

    st.info(
        "Historical traffic analytics can help identify recurring "
        "peak periods and support smarter traffic-management decisions."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.write("")
st.markdown("---")

st.markdown("### 🚦 SMART TRAFFIC AI")

st.caption(
    "AI-assisted traffic intelligence for smarter cities"
)

st.caption(
    "Traffic Monitoring • Prediction • Analytics • Smart Mobility"
)