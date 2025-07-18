# AICTExShell_week1_fireclassification

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

## 🧪 How to Run

### Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
