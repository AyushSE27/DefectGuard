# 🛡️ AI Vision DefectGuard
> **Low-Cost Automated Quality Control & Defect Detection System for MSMEs using YOLOv8 & Computer Vision**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://defectguard.streamlit.app/)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![YOLOv8](https://img.shields.io/badge/AI%20Model-YOLOv8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Problem Statement
Small and Medium Enterprises (MSMEs) in manufacturing face high product rejection rates and revenue losses due to manual quality inspection errors. Existing industrial computer vision setups are extremely expensive ($5,000–$20,000+), requiring specialized hardware and high-end sensors that small manufacturing units cannot afford.

## 🚀 Solution
**AI Vision DefectGuard** is a software-first, low-cost AI quality assurance platform that turns any standard laptop webcam, smartphone camera, or entry-level IP camera into a real-time automated inspection station.

Powered by **YOLOv8 Neural Networks** and OpenCV, it automatically detects surface anomalies, cracks, scratches, and structural defects in real-time with **zero additional hardware investments**.

---

## ✨ Key Features

- **🤖 Dual AI Inspection Modes:**
  - **Live Camera Stream (Localhost):** Real-time video inspection for factory assembly lines.
  - **Image Upload Inspection (Cloud Mode):** Instant batch/single product testing via cloud link.
- **⚡ Ultra-Fast Anomaly Detection:** Powered by Ultralytics YOLOv8 nano model for real-time inference without needing heavy GPUs.
- **📊 Real-time Visual Analytics:** Interactive dashboard displaying live Passed vs. Rejected metrics, Pass/Fail ratios, and pie-chart analytics using Plotly.
- **💡 Adjustable Sensitivity & Confidence Thresholds:** Fine-tune AI detection thresholds dynamically via the sidebar to avoid false positives.
- **💰 100% Hardware Agnostic:** Works on existing devices (Laptops, Webcams, Smartphones) with zero deployment friction.

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3.10+
- **Computer Vision & AI:** Ultralytics YOLOv8, OpenCV (`opencv-python-headless`)
- **Frontend & Dashboard:** Streamlit
- **Data Visualization:** Plotly Express, Pandas
- **Image Processing:** PIL (Python Imaging Library)

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/AyushSe27/AI-Vision-DefectGuard.git](https://github.com/AyushSE27/AI-Vision-DefectGuard.git)
cd DefectGuard