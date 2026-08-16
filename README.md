# 🚦 Smart Traffic AI

Smart Traffic AI is an AI-powered traffic prediction and analytics application built using Python, Machine Learning, Pandas, and Streamlit.

The application predicts traffic volume based on selected conditions and provides interactive traffic analytics and congestion information.

## ✨ Features

* 🚦 AI-based traffic volume prediction
* 📊 Interactive traffic analytics
* 📈 Average traffic by hour
* 🟢 Traffic congestion level detection
* 📅 Traffic pattern analysis
* 🖥️ Interactive Streamlit dashboard
* 🤖 Machine Learning model for traffic prediction

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib

## 📂 Project Structure

```text
Smart_Traffic_Ai/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── traffic_analysis.jpeg
│   ├── traffic_hero.jpeg
│   └── traffic_signal.jpeg
│
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
│
└── model/
    └── traffic_model.pkl
```

## ⚠️ Model File

The trained `traffic_model.pkl` file is not included in the GitHub repository because of its large file size.

To run the application locally, the trained model must be present at:

```text
model/traffic_model.pkl
```

## ▶️ How to Run the Application Locally

### 1. Download the Project

Download the repository from GitHub or clone it using:

```bash
git clone https://github.com/riyasharmaa30/Smart_Traffic_AI.git
```

### 2. Open the Project in VS Code

Open the downloaded `Smart_Traffic_Ai` folder in Visual Studio Code.

### 3. Open the Terminal

In VS Code, open:

**Terminal → New Terminal**

Make sure the terminal is inside the project folder.

### 4. Install the Required Libraries

Run:

```bash
python -m pip install -r requirements.txt
```

This installs all the Python libraries required by the application.

### 5. Make Sure the Model File Exists

Check that the trained model is available here:

```text
model/traffic_model.pkl
```

The application needs this file to make traffic predictions.

### 6. Start the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

### 7. Open the Application

After starting Streamlit, the terminal will display a local address similar to:

```text
Local URL: http://localhost:8501
```

Open the **Local URL** in your web browser.

The Smart Traffic AI dashboard should then open.

## 🧠 How It Works

1. The user enters/selects the required traffic conditions.
2. The trained Machine Learning model processes the input.
3. The application predicts the expected traffic volume.
4. The prediction is converted into a traffic congestion level.
5. The dashboard displays the prediction and traffic analytics.

## 📊 Dataset

The project uses the **Metro Interstate Traffic Volume** dataset for traffic analysis and prediction.

## 👩‍💻 Author

**Riya Sharma**

## 📌 Important Note

This project is currently designed to run **locally using Streamlit**.

GitHub is used to store and share the project source code. The Streamlit application itself is not automatically executed by GitHub.

│   └── features.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
