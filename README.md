# Adaptive Learning Engine (Bayesian Knowledge Tracing)

## Overview
This repository contains a full-stack, data-driven EdTech project that simulates an **Adaptive Learning Engine**. It utilizes a **Bayesian Knowledge Tracing (BKT)** model to predict a student's hidden knowledge state in real-time. 

Built with the rigor of data engineering and the pedagogical insight of over 20 years in international math education (IB DP & SAT), this engine demonstrates how to scale personalized tutoring by mathematically determining exactly when a student has mastered a concept.

## Business Impact
One of the most expensive and complex bottlenecks in digital education is identifying the moment a student is ready to advance. This model solves that by:
*   **Preventing Student Burnout:** Automatically recommending when to stop practicing a topic if the student is consistently frustrated.
*   **Optimizing Learning Paths:** Detecting mastery the moment the probability crosses the 85% threshold, eliminating redundant exercises.
*   **Translating Data into Pedagogy:** Capturing human behaviors mathematically, such as "careless errors" (Slip) and "lucky guesses" (Guess), rather than just grading raw final answers.

## Tech Stack & Methodology
*   **Data Generation:** Python (`pandas`, `numpy`) using Hidden Markov Models (HMM) to simulate realistic learning curves, aptitude variance, and chronological interactions.
*   **Machine Learning (Modeling):** `pyBKT` for training the probabilistic model, applying Cross-Validation, and calculating AUC/RMSE metrics.
*   **MLOps & Deployment:** `Streamlit` and `Plotly` for building a live, interactive web application that updates Bayesian probabilities on the fly without needing to reload the entire model in memory.

## Repository Structure
*   `generar_datos.py`: Script to generate a synthetic, highly realistic transactional dataset of student interactions based on SAT and IB Math curricula.
*   `entrenar_modelo.py`: Script to load the dataset, fit the BKT model, evaluate it (AUC/RMSE), and extract the core learning parameters.
*   `synthetic_bkt_dataset.csv`: The generated historical dataset (10,000+ interactions) used for training.
*   `app.py`: The main Streamlit application file containing the UI and the real-time Bayesian update logic.
*   `requirements.txt`: Dependencies required to run the web application.

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/wolayav/edtech-adaptive-learning.git
   cd edtech-adaptive-learning
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive simulation:
   ```bash
   streamlit run app.py
   ```

## Live Demo
Test the live interactive application here: **https://edtech-adaptive-learning.streamlit.app/**

---
*Developed by **Willy Olaya** | Data Scientist & EdTech Specialist*
