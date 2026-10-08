# Adaptive Edge-Cloud Inference Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PySpark](https://img.shields.io/badge/PySpark-%23E25A1C.svg?style=flat&logo=Apache-Spark&logoColor=white)](https://spark.apache.org/docs/latest/api/python/)

> **A self-adaptive, distributed streaming system for uncertainty-aware AI inference across the computing continuum.**

This project demonstrates dynamic resource allocation and reliability in next-generation AI-enabled systems. It processes **NASA Bearing Dataset** vibration telemetry using a PySpark pipeline and dynamically routes inference requests between a lightweight **Edge Model** and a heavy **Cloud Model** based on predictive uncertainty.

## 🌟 Key Features

* **Uncertainty-Aware Routing**: Calculates prediction entropy (augmented with an Out-Of-Distribution penalty) to detect uncertain or highly noisy samples and offload them to the cloud.
* **Self-Adaptive MAPE-K Loop**: Continuously monitors model uncertainty and cloud server capacity, dynamically adjusting the offloading threshold in real-time.
* **Computing Continuum Simulation**: Integrates PyTorch edge models with full-precision cloud models.
* **Streaming Data Processing**: Utilizes PySpark Structured Streaming (optimized with PyArrow) for robust, scalable data ingestion.

## 🏗️ Architecture Explained

1. **Stream (Data)**: High-frequency NASA Bearing vibration data (simulating progressive wear and tear) is streamed into the system via a socket server.
2. **PySpark Pipeline**: Ingests and micro-batches the data stream.
3. **Edge Inference**: A lightweight PyTorch model processes all incoming data.
4. **Uncertainty Monitor**: Calculates the entropy of the Edge Model's output. As the physical bearing degrades, the vibration variance increases, causing uncertainty to spike.
5. **MAPE-K Controller**: 
    * **Monitor**: Tracks how many samples are offloaded.
    * **Analyze**: Determines if we are exceeding cloud capacity limits.
    * **Plan & Execute**: Dynamically raises or lowers the uncertainty threshold to throttle cloud usage.

## 🚀 How to Run the Project

### Phase 1: Real Model Training (Google Colab)
Instead of simulating weights, you can train real models on the 6.5 GB NASA Bearing dataset.
1. Open [Google Colab](https://colab.research.google.com/).
2. Copy the code from `NASA_Bearing_Training.py` into a new notebook and run it. 
3. *Note: It uses `kagglehub` to securely and automatically download the dataset—no API keys required!*
4. Once training completes, download `edge_model.pth` and `cloud_model.pth` and place them in your local `checkpoints/` folder.

### Phase 2: Local Environment Setup
Ensure you have Python installed, and Hadoop `winutils` configured for Windows PySpark. 
*By default, the script looks for Hadoop in `C:\Users\Lenovo\Desktop\MASTER\Projet`.*

```bash
# 1. Open a terminal in the project folder and create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Phase 3: Run the Live Pipeline
With your virtual environment activated, run the main entry point:
```bash
python main.py
```
**What you will see:**
* The simulator will start generating high-frequency NASA bearing vibration data.
* As the bearing's `wear level` increases over time, the data becomes chaotic.
* The Edge model will become uncertain and begin offloading samples to the Cloud.
* You will see the **MAPE-K Controller** actively adjust the threshold (e.g., `Threshold adjusted from 0.80 to 0.85`) to ensure the cloud isn't overwhelmed.

## 🔬 Motivation: REGAIN-AI
This project aligns directly with the core research areas of the **REGAIN-AI** initiative at DTU Compute, specifically addressing the characterization, bounding, and control of uncertainty introduced by AI components in resource-constrained environments.

## 📜 License
MIT License
