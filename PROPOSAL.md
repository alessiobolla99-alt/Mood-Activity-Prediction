# Mood & Activity Prediction

**Student:** Alessio Bolla  
**Institution:** HEC Lausanne  
**Date:** November 19, 2025

**Dataset:** Daylio Mood Tracker Dataset  
**Source:** https://www.kaggle.com/datasets/kingabzpro/daylio-mood-tracker/data

---

## Project Overview

The Daylio Mood Tracker dataset contains real human-reported mood labels from users of the Daylio mood tracking application. Participants self-reported their emotional state at different moments during the day, along with associated behavioral and lifestyle variables such as activities performed, sleep patterns, and daily routines. The labels are not generated automatically or rule-based; they come directly from user input, making this a valid supervised learning problem based on actual human behavioral patterns.

The goal of this project is to develop a predictive model that estimates daily mood or stress levels based on behavioral and lifestyle variables. The main objective is to explore how factors such as sleep duration, daily activities, hours of work or study, and social interaction relate to emotional well-being. These predictions could help individuals, mental health professionals, and researchers better understand the relationship between daily behaviors and emotional states, enabling data-driven approaches to mood regulation and mental health monitoring.

The project will involve the following main steps:

### 1. Data cleaning and preprocessing
Handling missing values, duplicates, and inconsistencies to ensure data reliability. Encoding categorical mood labels and activity types. Normalizing numerical features such as timestamps and activity counts.

### 2. Feature engineering
Extracting and creating meaningful variables from raw data. Temporal features will be engineered to capture both long-term mood patterns and potential daily or weekly cycles. Behavioral variables will be aggregated at the daily level to summarize activity intensity, social interaction frequency, and routine consistency. Categorical variables, such as activity types, will be transformed using encoding techniques to enable their effective use in the predictive model.

### 3. Data analysis EDA
Analyzing relationships between behavioral features and mood states through correlation matrices for continuous variables and time-based visualizations to detect cyclical patterns or evolving emotional trends. Distribution analysis of mood states across different time periods and activity contexts. Heatmaps and scatter plots to explore feature-mood relationships.

### 4. Model development and evaluation
Implementing and comparing 3 machine learning models: Random Forest, Gradient Boosting, and XGBoost, to predict mood states. **Temporal train/test split will be used:** predicting future days rather than random days to test real-world generalization. This approach ensures the model learns genuine temporal patterns rather than memorizing specific days.

### 5. Model interpretability
Using feature importance analysis and SHAP (SHapley Additive exPlanations) values to understand which lifestyle factors most influence mood predictions. Identifying key possible behavioral predictors of emotional well-being to provide actionable insights.

### 6. Performance assessment
Evaluating models using appropriate metrics such as accuracy, balanced accuracy, F1-score, and confusion matrix evaluation to select the most accurate and reliable model for mood prediction.

I have always been fascinated by the intersection of behavioral science and machine learning, particularly how data-driven approaches can reveal patterns in human emotional well-being. Understanding the relationship between daily activities and mood has important applications in personal mental health monitoring, preventive interventions, and lifestyle optimization. This dataset contains authentic self-reported mood data from real users, allowing the model to learn genuine human patterns rather than artificial labels.

By integrating temporal analysi and interpretability techniques, I aim to build an accurate predictive model. This project combines technical precision and practical relevance, demonstrating both expertise in machine learning and understanding of the industry in an area with significant social impact.