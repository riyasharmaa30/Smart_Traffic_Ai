# 🚦 Smart Traffic AI

An AI-powered traffic intelligence system that predicts traffic volume, analyzes congestion patterns, and provides smart traffic-management recommendations.

Smart Traffic AI combines **Machine Learning, traffic analytics, weather information, and AI-based vehicle detection** into an interactive Streamlit dashboard.

---

## 📌 Project Overview

Traffic congestion is a major challenge in modern cities. Smart Traffic AI aims to assist traffic authorities by analyzing historical traffic data and predicting traffic volume based on different environmental and time-related conditions.

The application provides an interactive dashboard where users can:

- Enter traffic and weather conditions
- Predict traffic volume using a trained Machine Learning model
- Understand the predicted congestion level
- Analyze historical traffic patterns
- Identify peak traffic hours
- Capture traffic scenes using a webcam
- Detect vehicles in captured traffic images using YOLO
- View AI-assisted recommendations for traffic management

---

## ✨ Key Features

### 🤖 AI Traffic Prediction
Predicts traffic volume using:

- Hour of the day
- Temperature
- Rain
- Snow
- Cloud coverage
- Weather condition

The prediction is classified into traffic levels such as:

- 🟢 LOW
- 🟡 MODERATE
- 🔴 HIGH / CRITICAL

---

### 📊 Traffic Analytics

The dashboard analyzes historical traffic data and provides:

- Average traffic by hour
- Traffic-volume visualization
- Peak traffic hour
- Peak traffic volume
- Congestion severity

---

### 📷 Live Camera Monitoring

The application includes a webcam-based traffic monitoring feature.

Users can capture a traffic scene directly through the application.

The captured image can then be analyzed for:

- Scene visibility
- Brightness
- Traffic conditions
- Vehicle presence

---

### 🚗 AI Vehicle Detection

Smart Traffic AI uses the **YOLO object-detection model** to identify vehicles in captured traffic scenes.

The system detects common vehicle classes such as:

- 🚗 Cars
- 🏍️ Motorcycles
- 🚌 Buses
- 🚚 Trucks

The detected vehicles are counted and displayed on the dashboard along with an annotated image showing the detected objects.

---

### 🧠 AI Traffic Recommendations

Based on predicted traffic conditions, the application provides recommendations such as:

- Traffic monitoring
- Signal optimization
- Congestion alerts
- Route management
- Deployment of traffic personnel

---

### 🏛️ Traffic Authority Action Center

The dashboard provides an AI-assisted action panel for traffic authorities, including:

- 🚦 Signal Optimization
- 📢 Congestion Alerts
- 🛣️ Route Management

These recommendations demonstrate how traffic prediction can support smarter city management.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data processing and analysis
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine Learning
- **Streamlit** – Interactive web application
- **OpenCV** – Image and camera processing
- **YOLO** – AI-based vehicle detection
- **Joblib** – Model loading
- **Matplotlib / Streamlit Charts** – Data visualization
- **Git & GitHub** – Version control

---

## 📂 Project Structure

```text
Smart_Traffic_Ai/
│
├── assets/
│   ├── traffic_hero.jpeg
│   ├── traffic_analysis.jpeg
│   └── traffic_signal.jpeg
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
│
├── model/
│   ├── traffic_model.pkl
│   └── features.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md