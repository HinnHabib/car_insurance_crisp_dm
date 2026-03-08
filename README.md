# Car Insurance Claim Prediction

A binary classification project that predicts whether a customer will file a car insurance claim based on demographics, vehicle information, and driving behavior. The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

## Overview

- **Task:** Binary classification (claim vs. no claim)
- **Dataset:** [Car Insurance Data](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data) from Kaggle
- **Best Model:** Random Forest (test accuracy ~0.84)

## Project Structure

```
car_insurance_crisp_dm/
├── car_insurance_crisp_dm.ipynb   # Main Jupyter notebook
└── README.md
```

## Dataset

The dataset includes 18 features such as:

| Feature | Description |
|---------|-------------|
| `age` | Driver's age |
| `gender` | Driver's gender |
| `driving_experience` | Years of driving (e.g., 0-9y, 10-19y) |
| `education` | Highest education level |
| `income` | Income bracket |
| `credit_score` | Normalized credit score (0–1) |
| `vehicle_ownership` | Owns vehicle (0/1) |
| `vehicle_year` | Before/After 2015 |
| `annual_mileage` | Estimated yearly miles driven |
| `speeding_violations` | Number of speeding tickets |
| `duis` | DUI offenses |
| `past_accidents` | Past accident count |
| `outcome` | **Target:** 1 = claim filed, 0 = no claim |

## CRISP-DM Phases

1. **Business Understanding** – Define classification goal and success criteria
2. **Data Understanding** – EDA, missing values, distributions, correlations
3. **Data Preparation** – Preprocessing, feature engineering (PCA), feature selection (collinearity filter)
4. **Modeling** – Default models, feature-engineered models, neural networks, hyperparameter tuning
5. **Evaluation** – Model comparison and business interpretation
6. **Deployment** – Recommendations for batch scoring and API integration

## Models Used

| Model | Description |
|-------|-------------|
| **Random Forest** | Default and with PCA + collinearity filter (best performer) |
| **Logistic Regression** | Baseline linear model |
| **K-Nearest Neighbors** | Instance-based classifier |
| **Neural Network** | 1 hidden layer, Keras |
| **Tuned Neural Network** | Keras Tuner (units, dropout, optimizer, learning rate) |

## Key Results

- **Random Forest:** Test accuracy ~0.84, precision/recall ~0.88 (no-claim) and ~0.74 (claim)
- **Feature engineering:** PCA (3 components) + collinearity filter keeps accuracy at 0.84 with fewer features
- **Neural networks:** ~0.83 test accuracy

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- tensorflow, keras
- keras_tuner
- collinearity

Install with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras_tuner collinearity
```

## Usage

1. Download the [Car Insurance Data](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data) from Kaggle.
2. Place the data file in your working directory (or update the path in the notebook).
3. Open `car_insurance_crisp_dm.ipynb` in Jupyter or Google Colab.
4. Run all cells. The notebook uses Google Drive mount for Colab; adjust paths if running locally.

## Deployment Recommendations

- **Batch scoring:** Score new applicants for underwriting
- **API:** Integrate into quote systems so agents see claim-risk scores in real time

## License

See the dataset source for data licensing. Code is provided as-is for educational use.
