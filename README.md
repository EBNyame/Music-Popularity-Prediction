# 🎵 Music Popularity Prediction using Random Forest

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview

Music popularity prediction involves using **machine learning regression techniques** to predict how popular a song will be based on its musical attributes.

In this project, we use a **Random Forest Regressor** with **GridSearchCV** to predict a song's popularity using features such as:

* Energy
* Valence
* Danceability
* Loudness
* Tempo
* Speechiness
* Liveness

The goal is to understand how musical characteristics influence popularity and to build a predictive model that can assist artists, producers, and marketers in decision-making.

---

## 📊 Dataset Description

* **Source:** Spotify music dataset
* **Number of tracks:** 227
* **Target variable:** `Popularity`
* **Feature types:** Numerical music attributes and metadata

### 🎯 What is Popularity?

`Popularity` is a numerical score (typically between **0–100**) representing how popular a track is on Spotify. It is calculated based on factors such as:

* Number of streams
* Recent listening trends
* User engagement

Higher values indicate more popular songs.

---

## ⚙️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🔍 Exploratory Data Analysis (EDA)

The following visualizations were used:

* **Scatter plots** to observe relationships between features and popularity
* **Histograms with KDE** to understand feature distributions
* **Correlation heatmap** to identify relationships between numerical features

These steps helped in feature understanding and selection.

---

## 🧠 Model Building

### 1️⃣ Feature Selection

```python
features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Tempo', 'Speechiness', 'Liveness']
X = spotify_data[features]
Y = spotify_data['Popularity']
```

### 2️⃣ Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### 3️⃣ Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4️⃣ Model & Hyperparameter Tuning

We used **GridSearchCV** to find the best parameters for the Random Forest Regressor.

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

```python
grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    verbose=2,
    refit=True
)

grid_search_rf.fit(X_train_scaled, Y_train)
```

---

## 📈 Model Evaluation

The model was evaluated using:

* **Mean Squared Error (MSE)**
* **R² Score**

These metrics help measure prediction accuracy and goodness of fit.

---

## 🚀 Results

* The tuned Random Forest model successfully learned patterns between music features and popularity
* Feature importance analysis can further explain which attributes influence popularity most

---

## 📁 Project Structure

```
├── data/
│   └── spotify_dataset.csv
├── notebooks/
│   └── music_popularity_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 🧪 Future Improvements

* Try other regression models (Linear Regression, XGBoost)
* Increase dataset size
* Add feature importance visualization
* Deploy as a web app

---

## 🧑‍💻 Author

**Exodus Blessed Nyame**
BSc Information Technology, University of Ghana
Aspiring Backend & Machine Learning Engineer

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you found this project helpful, feel free to star the repository!
