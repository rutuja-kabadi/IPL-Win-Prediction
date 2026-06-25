# 🏏 IPL Win Probability Dashboard

An end-to-end Machine Learning project that predicts the **real-time win probability** of the chasing team in an IPL match after every ball. The project leverages ball-by-ball IPL data, domain-specific feature engineering, and an XGBoost classifier to estimate winning chances dynamically throughout the second innings. Predictions are served through an interactive Streamlit dashboard, allowing users to simulate different match scenarios and visualize changing win probabilities.

## 🚀 Project Objectives

* Predict the probability of the chasing team winning after every delivery.
* Build a production-style ML pipeline from data preprocessing to deployment.
* Engineer cricket-specific features that capture the state and momentum of a match.
* Deploy an interactive dashboard for real-time predictions and scenario analysis.

---

## 📊 Dataset

The project uses the IPL Ball-by-Ball Dataset containing over **700,000+ deliveries** across multiple IPL seasons.

**Files used**

* `matches.csv` – Match metadata, teams, venue, winner
* `deliveries.csv` – Ball-by-ball match events

---

## 🛠️ Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Cricket-specific Feature Engineering
4. Leak-free Train/Test Split (Match-level)
5. Model Training using XGBoost
6. Model Evaluation
7. Interactive Streamlit Dashboard

---

## ⚙️ Feature Engineering

The model uses both match-state and momentum-based features, including:

* Runs Scored
* Target Score
* Runs Remaining
* Balls Remaining
* Wickets Fallen
* Wickets in Hand
* Current Run Rate (CRR)
* Required Run Rate (RRR)
* Run Rate Difference
* Rolling Momentum (Last 6 & 12 Balls)
* Partnership Runs
* Match Phase (Powerplay, Middle Overs, Death Overs)
* Team Encodings
* Target Buckets

---

## 🤖 Machine Learning

**Model:** XGBoost Classifier

The model predicts the probability that the chasing team eventually wins the match based on the current match situation.

To ensure realistic evaluation, the dataset is split **by match ID**, preventing data leakage between training and testing sets.

---

## 📈 Model Evaluation

Evaluation metrics include:

* ROC-AUC
* Log Loss
* Brier Score (Probability Calibration)

Approximate performance:

* **ROC-AUC:** ~0.85
* **Log Loss:** ~0.40
* **Brier Score:** ~0.16

---

## 📊 Streamlit Dashboard

The interactive dashboard allows users to:

* Select batting and bowling teams
* Enter target score
* Adjust current score, wickets, and balls played
* View real-time win probability
* Visualize probability trends using Plotly
* Explore "what-if" match scenarios
* Analyze key match statistics

---

## 💻 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Plotly
* Pickle

---

## 📁 Project Structure

```
IPL-Win-Probability-Dashboard/
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
├── notebooks/
│   └── feature_engineering.ipynb
│
├── model/
│   ├── ipl_model.pkl
│   ├── le_bat.pkl
│   └── le_bowl.pkl
│
├── app.py
├── requirements.txt
├── README.md
```

---

## ✨ Key Highlights

* Processed **700K+** IPL ball-by-ball records.
* Engineered **20+** domain-specific cricket features.
* Implemented a **leak-free match-level train/test split** for robust evaluation.
* Built a calibrated probability prediction model using **XGBoost**.
* Developed an interactive **Streamlit** dashboard for real-time match simulation and visualization.

---

## 🎯 Future Improvements

* Incorporate player-level statistics and recent form.
* Add venue and weather conditions.
* Use SHAP for explainable AI.
* Integrate live IPL match data for real-time predictions.
