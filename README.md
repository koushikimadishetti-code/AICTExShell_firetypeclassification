# AICTExShellxEdunetFoundation_fireclassification
🔥 Classification of Fire Types in India Using MODIS Satellite Data(Deforestation Detection).

This repository contains the complete workflow for building and deploying a machine learning model to classify fire types in India using MODIS satellite data provided by NASA.

📁 Repository Contents

Classification_of_Fire_Types_in_India_Using_MODIS_Satellite_Data_koushiki.ipynb

Jupyter notebook containing the full code for preprocessing, model training, and evaluation.

Datasets Used for Training:

MODIS_NASA_INDIA_2021_dataset

MODIS_NASA_INDIA_2022_dataset

MODIS_NASA_INDIA_2023_dataset

app.py
Streamlit-based frontend application that allows users to predict fire types based on satellite readings using the best-trained model.

scaler.pkl
Serialized Scikit-learn scaler used to preprocess user input before prediction.

🚀 Deployment
The frontend is built using Streamlit. Run the app with the following command:
streamlit run app.py
Note:Make sure scaler.pkl and the model file are in the same directory as app.py.

🧠 Best Model
The best-trained model is not included in the repository due to its large size. You can download it using the link below:

👉https://drive.google.com/file/d/1G4i76hv1hauLdDbdld-kgdW7GedNdfzY/view?usp=sharing

## 🧪 How to Run
▶️ Step 1:Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost statsmodels imbalanced-learn folium


▶️ Step 2: Run the Streamlit App
Once the dependencies are installed, you can launch the web application using:

streamlit run app.py

Note: Make sure the following files are in the same directory:

->app.py

->best_fire_detection_model.pkl

->scaler.pkl



# AICTExShellxEdunetFoundation_fireclassification
# 🔥 Fire Type Classification - Week 1 ML Project

This project is an introductory machine learning exercise involving data preprocessing, exploration, and visualization using fire incident data. It combines multiple datasets and investigates the distribution of various features to prepare for classification tasks.

---

## 📁 Datasets
- `dataset1.csv`
- `dataset2.csv`
- `dataset3.csv`

These datasets are merged into a single DataFrame for unified analysis.

---

## 📌 Key Steps

### ✅ Libraries Used
- `numpy`, `pandas` – data manipulation
- `matplotlib`, `seaborn` – data visualization
- `scikit-learn` – model building & preprocessing
- `xgboost` – advanced classification model

---

### 📊 Data Preparation & Exploration
- Loaded three datasets and combined them
- Checked for:
  - Missing values
  - Duplicates
  - Data types
- Explored basic statistics using `.describe()`
- Examined class distribution in the target variable (`type`)
- Investigated categorical columns for uniqueness

---

### 📈 Visualizations
- **Class Balance**: Count plot of fire `type` using `sns.countplot()`
- **Confidence Distribution**: Histogram + KDE of the `confidence` feature

---


# 🔥 Fire Type Classification - Week 2 ML Project: Advanced Preprocessing & Spatial Analysis
This project builds upon Week 1's foundational data exploration by implementing more advanced data preprocessing techniques, including detailed distribution analysis, outlier treatment, temporal feature engineering, categorical encoding, numerical scaling, and crucial handling of imbalanced datasets using SMOTE. It also introduces spatial visualization with Folium.


## 📌 Key Steps
✅ Libraries Used
numpy, pandas – Data manipulation and analysis

matplotlib, seaborn – Comprehensive data visualization

scikit-learn – Data preprocessing (scaling, feature selection)

xgboost – Advanced classification model (though not explicitly used for training in the provided snippets, it's imported)

statsmodels, scipy.stats – Statistical modeling and distribution analysis (KDE, QQ plots)

imblearn – Handling imbalanced datasets (SMOTE)

folium – Interactive geographical mapping

## 📊 Data Preparation & Exploration
Initial Data Checks:

Loaded three datasets and combined them.

Checked for missing values, duplicates, and data types.

Explored basic statistics using .describe().

Examined class distribution in the target variable (type).

Investigated categorical columns for uniqueness.

Detailed Numerical Feature Analysis:

Generated histograms for key numerical features.

Performed Kernel Density Estimation (KDE) plots to visualize the probability density of numerical features.

Created Quantile-Quantile (QQ) plots to assess if numerical features follow a normal distribution.

Temporal Feature Engineering:

Converted acq_date to datetime objects.

Extracted year, month, day_of_week, day_of_year, and hour to capture temporal patterns.

Outlier Treatment:

Visualized outliers using box plots for key numerical features.

Implemented and applied an IQR (Interquartile Range) based method to remove outliers from numerical columns.

Re-visualized box plots to confirm outlier reduction.

Categorical Encoding:

Applied One-Hot Encoding (pd.get_dummies) to categorical features (daynight, satellite, instrument) to convert them into a numerical format suitable for machine learning models.

Numerical Feature Scaling:

Used StandardScaler to normalize continuous numerical variables (brightness, scan, track, confidence, bright_t31, frp), ensuring they contribute equally to model training.

Feature and Target Separation:

Separated the preprocessed features (X) from the target variable (y, which is 'type').

## 📈 Visualizations
Class Balance: Count plot of fire type using sns.countplot().

Confidence Distribution: Box plot of confidence by fire type using sns.boxplot().

Fire Locations by Type: Scatter plot of latitude vs. longitude colored by fire type using sns.scatterplot().

Distribution of Day/Night Observations: Count plot for daynight using sns.countplot().

Distribution of Satellite Observations: Count plot for satellite using sns.countplot().

Distribution of Version: Count plot for version using sns.countplot().

Correlation Heatmap: Visualized the correlation matrix of numerical features (latitude, longitude, brightness, confidence, frp) using sns.heatmap().

Histograms of Numerical Features: Displayed distributions of key numerical features.

KDE Plots: Visualized the density distribution of brightness, confidence, frp, bright_t31, scan, track.

QQ Plots: Assessed the normality of brightness, confidence, frp, bright_t31, scan, track.

Fire Detections by Month: Count plot showing the frequency of fire detections across different months.

Box Plots for Outliers: Visualized numerical features' distributions before and after outlier removal.

Interactive Fire Map: Used folium to create an interactive map of India, plotting sampled fire locations with popups displaying FRP and acquisition date.

## ⚖️ Imbalanced Data Handling
SMOTE (Synthetic Minority Over-sampling Technique): Applied SMOTE to the training data (X, y) to synthesize new examples for minority classes. This balances the class distribution, which is crucial for training robust machine learning models on imbalanced datasets. The distribution of the target variable after SMOTE is printed to confirm the balancing.


# 🔥 Fire Type Classification - Week 3 work for this ML Project
In Week 3, I focused on building, evaluating, and deploying the machine learning model for fire type classification. The key tasks completed include:
🔄 Data Splitting:

Split the resampled dataset into training and testing sets using train_test_split with stratification to preserve class balance.

🧠 Model Training & Evaluation:

Trained and evaluated multiple classification algorithms:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

K-Nearest Neighbors Classifier

Evaluated models using:

Accuracy Score

Classification Report

Confusion Matrix

🏆 Model Selection:

Random Forest was identified as the best-performing model based on accuracy.

💾 Model & Scaler Export:

Saved the best-trained model as best_fire_detection_model.pkl.

Saved the preprocessing StandardScaler as scaler.pkl.

🌐 Streamlit App Development:

Created a user-friendly Streamlit web application (app.py) for deploying the model.

The app takes six input features from MODIS satellite data and predicts the fire type using the trained model.
This concludes a full cycle from model training to basic deployment.

