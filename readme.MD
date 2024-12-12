# Evomo Power Smart Detection Data Preprocessing

**Bangkit Company Capstone Project [C242-PS01]**

## Table of Contents
- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Problem Statements](#problem-statements)
- [Project Goals & Objectives](#project-goals--objectives)
- [Solution Approach](#solution-approach)
- [Data Description](#data-description)
- [Model Development](#model-development)
  - [Preprocessing Steps](#preprocessing-steps)
  - [Anomaly Detection Models](#anomaly-detection-models)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Usage Examples](#usage-examples)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project focuses on energy usage anomaly detection within buildings. By leveraging machine learning models, we aim to detect unusual patterns or spikes in electricity consumption across different facility assets, such as Air Handling Units (AHUs), chillers, and lifts—enabling facility managers to identify and address potential issues before they become significant problems.

## Team Members
- **M011B4KY1007** - Darren Christian Liharja  
- **C002B4KY2028** - Ivanoel Rhema Christama Adhiprama  
- **C002B4KY3771** - Reyhan Ghifari Tanjung  
- **A001B4KY2884** - Muhammad Haris Sabil Al Karim  
- **A318B4KX1511** - Fina Febrianti

## Problem Statements
Inefficient and excessive energy consumption often goes unnoticed, leading to higher costs and negative environmental impacts. Without a proper detection system, patterns of abnormal energy usage remain hidden, resulting in unnecessary waste and potential equipment damage. This project seeks to implement a comprehensive solution that:
- Continuously monitors energy usage
- Identifies anomalies or irregularities in consumption patterns
- Alerts facility managers to take timely, corrective actions

## Project Goals & Objectives
- **Anomaly Detection**: Promptly identify spikes in energy consumption associated with mechanical failures, suboptimal settings, or operational inefficiencies.
- **Real-Time Monitoring**: Provide immediate insights into building energy usage to support informed decision-making.
- **Sustainability**: Reduce operational costs and carbon footprints by eliminating unnecessary energy waste.
- **Scalability**: Develop a solution that can easily integrate with different building management systems and handle varying data volumes.

## Solution Approach
1. **Data Acquisition & Cleaning**: Collect real-time and historical energy consumption data for various building assets.
2. **Feature Engineering & Preprocessing**: Clean, normalize, and structure data to ensure reliable model inputs.
3. **Modeling & Anomaly Detection**: Experiment with and implement anomaly detection techniques (Isolation Forest, LSTM Autoencoders, etc.).
4. **Deployment**: Integrate the model into a production environment for continuous monitoring, and provide a REST API for data querying and anomaly alerts.
5. **Visualization & Reporting**: Create user friendly dashboards and reports that highlight real-time anomalies and provide actionable recommendations.

## Data Description
The `data` directory contains a mix of CSV and Excel files representing historical and processed energy consumption data from various sources:
- **AHU_Lantai_2.csv**: Energy consumption data from Air Handling Unit on floor 2.
- **Lift_OPMC.csv / Lift_Witel_Jaksel.csv**: Lift energy usage and operational logs.
- **Chiller_Witel_Jaksel.csv**: Chiller energy usage for Wisma Telkom in South Jakarta

## Model Development
### Preprocessing Steps
- **Missing Value Handling**: Interpolate or impute missing measurements.
- **Normalization/Scaling**: Apply min-max normalization or standard scaling for improved model performance.
- **Feature Engineering**: Generate derived features such as rolling averages, lag features, or time based indicators.

### Anomaly Detection Models
- **Isolation Forest**: A tree-based anomaly detection method that isolates outliers by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
- **LSTM Forecasting**: Long Short-Term Memory (LSTM) networks can be trained on historical time series data to predict future energy consumption values.
- **Autoencoder Anomaly Detection**: An autoencoder is a neural network designed to learn a compressed representation of normal data and then reconstruct it with minimal loss.

## Project Structure
```
ANOMALY-DETECTION/
│
├─ data/
│  ├─ AHU_Lantai_2.csv
│  ├─ chiller_cleaned.xlsx
│  ├─ Chiller_Witel_Jaksel.csv
│  ├─ chiller.xlsx
│  ├─ conv_chiller.csv
│  ├─ conv_data.csv
│  ├─ conv_lift.csv
│  ├─ data_predicted.csv
│  ├─ Historical Data.xlsx
│  ├─ Lift_OPMC.csv
│  ├─ Lift_Witel_Jaksel.csv
│  └─ lift.xlsx
│
├─ deployment/
│  ├─ AHU_2.pkl
│  ├─ api_model.py
│  ├─ chiller.pkl
│  ├─ lift_opmc.pkl
│  ├─ lift_witel.pkl
│  └─ requirements.txt
│
├─ images/
│  ├─ autoencoder.png
│  ├─ isolation.png
│  ├─ LSTM Autoencoder.jpg
│  ├─ LSTM.jpg
│  └─ power.jpg
│
├─ .gitignore
├─ autoencoder.ipynb
├─ forecasting.ipynb
├─ isolation_forest.ipynb
├─ preprocess.ipynb
├─ readme.md
└─ requirements.txt
```

## Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Evomo-PowerSmart/Anomaly-Detection.git
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
- **Preprocessing & Model Training**:  
  Use the provided Jupyter notebooks (`preprocess.ipynb`, `isolation_forest.ipynb`, `autoencoder.ipynb`) to preprocess data, train, and evaluate models.
  
- **API Deployment**:  
  Navigate to the `deployment` folder and run:
  ```bash
  python api_model.py
  ```
  This will start a local server for inference. Adjust the code and configuration as needed before production deployment.

## Usage Examples
- **Detecting Anomalies**:  
  After running the API server, send a POST request with new energy usage data to the endpoint (e.g., `http://localhost:5000/predict_ahu`):
  ```bash
  curl -X POST http://localhost:5000/predict_ahu \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01 10:30:00", "usage": 150.0}'
  ```
  The response will indicate whether the input data points are anomalous.

## Future Work
- **Feature Improvements**: Incorporate additional contextual data (weather, occupancy, equipment maintenance logs) to improve model accuracy.
- **Model Optimization**: Experiment with other anomaly detection methods (Prophet-based forecasting, hybrid models) to enhance performance.
- **Scalability**: Deploy the solution on cloud platforms (Google Cloud, AWS) and integrate advanced monitoring and alerting tools.


## Acknowledgments
- **Bangkit Program** for providing the capstone opportunity.
- **Mentors and Instructors** for guidance and feedback.


