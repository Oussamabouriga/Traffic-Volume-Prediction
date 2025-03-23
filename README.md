# 🚦 Traffic Volume Prediction with PyTorch

This project builds a deep learning model using an **LSTM (Long Short-Term Memory)** network in **PyTorch** to predict traffic volume on a major highway. It uses a rich, time-series dataset from the UCI Machine Learning Repository, combining weather data, time features, and historical traffic data.

## 📊 Problem Overview

The goal is to forecast **future traffic volume** using historical and contextual information such as weather, date/time, and holiday indicators. This is a **regression problem** since traffic volume is a continuous numeric value.

## 📁 Dataset Description

The dataset has been preprocessed (normalized and encoded) and split into:
- `train_scaled.csv`
- `test_scaled.csv`

It includes numeric and categorical features representing weather, time, and traffic information.

| **Column**                  | **Type**       | **Description**                                                                 |
|----------------------------|----------------|---------------------------------------------------------------------------------|
| `temp`                     | Numeric        | Average temperature in Kelvin                                                  |
| `rain_1h`                  | Numeric        | Rain amount in mm for the hour                                                |
| `snow_1h`                  | Numeric        | Snow amount in mm for the hour                                                |
| `clouds_all`               | Numeric        | Cloud coverage in percentage                                                  |
| `date_time`                | DateTime       | Timestamp of data collection (local CST time)                                 |
| `holiday_*` (11 columns)   | Categorical    | US national + regional holidays, one-hot encoded                              |
| `weather_main_*` (11 cols) | Categorical    | Short weather description, one-hot encoded                                    |
| `weather_description_*` (35 cols) | Categorical | Longer weather descriptions, one-hot encoded                         |
| `traffic_volume`           | Numeric        | Hourly traffic count on I-94 highway                                          |
| `hour_of_day`              | Numeric        | Hour of the day (0–23)                                                        |
| `day_of_week`              | Numeric        | Day of the week (0 = Monday, 6 = Sunday)                                      |
| `day_of_month`             | Numeric        | Day number in the month                                                       |
| `month`                    | Numeric        | Month number (1 = January, 12 = December)                                     |

> **Target Variable**: `traffic_volume`

## 🧠 Model Architecture

| Layer              | Details                                        |
|-------------------|------------------------------------------------|
| LSTM              | 2 layers, hidden size = 64, batch_first=True   |
| Fully Connected   | `nn.Linear(hidden_size, 1)`                    |
| Activation        | `nn.LeakyReLU()`                               |

## ⚙️ Workflow

### 1️⃣ Data Preparation
- Split into sequences of length 24
- Target is traffic volume at the next hour
- Batched with `DataLoader`

### 2️⃣ Model Training
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (`lr=0.001`)
- **Epochs**: 2 (can increase to improve performance)

### 3️⃣ Model Evaluation
Evaluated on test set using:
- **MSE**
- **MAE**
- **R² Score**
- **Custom Accuracy**: % of predictions within ±10 units of the actual value

## ✅ Example Results

 - Epoch 1/2, Loss: 0.0004 Epoch 2/2, Loss: 0.0003

 - Test MSE: 0.0001 MAE: 0.0058 R² Score: 0.92 Custom Accuracy (±10): 98.74%


## 🛠️ How to Run

1. Clone this repo or open the notebook in Google Colab
2. Make sure `train_scaled.csv` and `test_scaled.csv` are present
3. Run all code cells in order
4. Observe the training loss and evaluation metrics

## 📚 Acknowledgements

Dataset: [UCI Machine Learning Repository - Metro Interstate Traffic Volume](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

