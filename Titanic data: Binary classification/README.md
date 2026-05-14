# Titanic Survival Prediction: Binary Classification

This project uses Machine Learning to predict the survival of passengers on the Titanic. The implementation follows a standard data science workflow, including Exploratory Data Analysis (EDA), data cleaning, and predictive modeling using Logistic Regression.

## 🚀 Project Overview
The goal is to determine which factors (like age, gender, or class) were most likely to lead to survival during the Titanic disaster. 
 
## 🛠️ Key Features
* **Data Analysis:** Comprehensive EDA using Seaborn and Matplotlib to visualize survival trends.
* **Data Preprocessing:** * Handled missing values using `SimpleImputer`.
    * Encoded categorical data with `LabelEncoder`.
    * Dropped irrelevant features (Names, Passenger IDs, Tickets, and Cabins).
* **Modeling:** Built a **Logistic Regression** model to classify passengers.
* **Evaluation:** Detailed performance tracking using Confusion Matrices and Classification Reports (Accuracy, Precision, Recall, and F1-Score).

## 📊 Technologies Used
* **Python** (Core Language)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Data Visualization)
* **Scikit-Learn** (Machine Learning)

## 📁 Dataset
The project uses the `tested.csv` file, which includes passenger demographics and survival status.

## 📈 Model Performance
The model provides a balanced prediction, evaluated through:
* **Confusion Matrix:** To visualize true positives vs. false positives.
* **F1-Score:** To ensure the model handles survival and non-survival predictions effectively.

---
*Created as part of my **Machine-Learning-Projects** portfolio.*
