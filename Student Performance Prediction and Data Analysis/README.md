
# Student Performance Analysis and Prediction

This repository contains a Data Science and Machine Learning project implemented in a **Jupyter Notebook**. The project focuses on Exploratory Data Analysis (EDA) and the implementation of classification models to predict student academic performance.

## Project Overview
The goal of this project is to analyze demographic, social, and academic factors that influence student grades. By processing the `Student_performance_data_.csv` dataset, the notebook identifies key correlations and trains machine learning models to classify students into "Pass" or "Fail" categories.

## Key Features

### 1. Data Preprocessing & Cleaning
* **Automated Parsing:** Utilizes the `csv` library's `Sniffer` to automatically detect file delimiters, ensuring robust data loading.
* **Data Integrity:** Checks for null values and strictly types columns.
* **Feature Engineering:** Converts the multi-class `GradeClass` variable into a binary target variable `pass_fail` (0 for Fail, 1 for Pass) to simplify the classification task.

### 2. Exploratory Data Analysis (EDA)
* **Statistical Analysis:** Generates descriptive statistics to understand the spread and central tendency of the data.
* **Visualization:** Uses **Seaborn** and **Matplotlib** to create:
    * **Distribution Histograms:** For numeric variables like `Age`, `StudyTimeWeekly`, and `Absences`.
    * **Correlation Heatmaps:** Pearson and Spearman matrices to identify relationships between features (e.g., the negative correlation between `Absences` and grades).
    * **Box Plots:** To visualize the impact of features like `StudyTimeWeekly` and `ParentalSupport` on the pass/fail rate.
    * **Count Plots:** To analyze the balance of categorical variables.

### 3. Machine Learning Models
The project implements and compares several classification algorithms (imported via `sklearn`) to predict student outcomes:
* **Logistic Regression**
* **Gaussian Naive Bayes**
* **Support Vector Machines (SVC)**
* **Decision Trees & Random Forest**
* **Principal Component Analysis (PCA):** Used for dimensionality reduction to improve model performance.

##Tech Stack
* **Language:** Python 
* **Libraries:**
    * `Pandas` & `NumPy`: Data manipulation and linear algebra.
    * `Matplotlib` & `Seaborn`: Data visualization.
    * `Scikit-learn`: Model training, evaluation, and preprocessing (`StandardScaler`, `train_test_split`).
