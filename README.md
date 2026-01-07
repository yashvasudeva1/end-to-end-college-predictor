# JoSAA College Predictor

A **data-driven, machine learning–based JoSAA college prediction system** that analyzes historical counselling data (2021–2025) to predict opening and closing ranks, visualize cutoff trends, and estimate admission chances for aspirants.

This project focuses on **decision support**, combining exploratory data analysis, predictive modeling, and interactive visualization using Streamlit.

---

## Motivation

JoSAA counselling is a complex, multi-round process where:
- Cutoffs vary significantly across rounds
- Competition changes year by year
- Branch demand often outweighs institute reputation

Most existing predictors rely on static last-year cutoffs and ignore round-wise and temporal behavior.  
This project addresses those gaps by learning **historical trends** and presenting them in an interpretable way.

---

## Key Features

### Exploratory Data Analysis (EDA)
- Year-wise competition trends
- Round-wise cutoff relaxation behavior
- Institute and branch competitiveness
- Cutoff stability and volatility analysis
- Clear, written conclusions derived from data

### Trend Visualizations
- Closing rank vs **Rounds (X-axis)** with **Year as hue**
- Year-wise cutoff movement analysis
- Interactive filters for institute and branch

### Machine Learning Models
- Separate models for:
  - Opening Rank Prediction
  - Closing Rank Prediction
- Trained on JoSAA data from 2021–2025
- Learns temporal and round-based cutoff patterns

### College Chance Estimation
- User inputs their JEE rank
- Model predicts future cutoffs
- Colleges grouped into:
  - Safe
  - Moderate
  - Risky
  - Very Risky
- All colleges are shown for transparency

### Model Performance Evaluation
- MAE, RMSE, R² metrics
- Actual vs Predicted plots
- Error distribution analysis
- Clear interpretation of model strengths and limitations

### Interactive Streamlit Dashboard
- Multi-page layout
- Tab-based EDA
- Clean, professional UI
- Clear separation between analysis, prediction, and evaluation

---

## Project Architecture
```
College Predictor/
│
├── data/
│ ├── raw/ # Raw JoSAA round-wise CSVs
│ ├── processed/ # Year-wise master datasets
│ └── final/
│ ├── final_jossa_dataset.csv
│ └── jossa_features.csv
│
├── utils/
│ └── data_preprocessing.py
│
├── features/
│ └── build_features.py
│
├── training/
│ ├── train_opening.py
│ └── train_closing.py
│
├── models/
│ ├── opening_rank_model.pkl
│ ├── closing_rank_model.pkl
│ └── encoders.pkl
│
├── app.py # Streamlit application
├── requirements.txt
└── README.md
```

---

## Data Pipeline

1. **Raw Data Ingestion**
   - JoSAA round-wise CSVs (2021–2025)

2. **Year-wise Master Creation**
   - Adds year and round columns
   - Standardizes schema across files

3. **Data Preprocessing**
   - Cleans rank values
   - Removes inconsistencies
   - Encodes categorical features

4. **Feature Engineering**
   - Lag features
   - Rolling statistics
   - Temporal indicators

5. **Model Training**
   - XGBoost regressors
   - Separate targets for opening and closing ranks

6. **Visualization & Prediction**
   - Streamlit dashboard
   - EDA, prediction, and performance evaluation

---

## Machine Learning Details

### Models Used
- **XGBoost Regressor**
  - Handles non-linear relationships
  - Robust to skewed rank distributions
  - Well-suited for tabular data

### Targets
- `opening_rank`
- `closing_rank`

### Why Two Models?
Opening and closing ranks exhibit different behaviors across rounds and years.  
Training separate models improves accuracy and interpretability.

---

## Evaluation Metrics

- **MAE (Mean Absolute Error)**  
  Average deviation from actual cutoffs

- **RMSE (Root Mean Squared Error)**  
  Penalizes large prediction errors

- **R² Score**  
  Measures how much variance is explained

> The model is intended for **relative comparison and trend-based guidance**, not exact rank guarantees.

---

## Disclaimer

- This project does **not** guarantee admission
- JoSAA policies and seat matrices may change
- Predictions should be used as **decision support**, not as final authority

---

## How to Run the Project

### Clone the repository
```
git clone https://github.com/your-username/College-Predictor.git
cd College-Predictor
```
### Install dependencies
```
pip install -r requirements.txt
```
### Run preprocessing and training (one-time)
```
python utils/data_preprocessing.py
python features/build_features.py
python training/train_opening.py
python training/train_closing.py
```

### Launch the Streamlit app
```
streamlit run app.py
```
