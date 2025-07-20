# Energy Consumption Analysis - Machine Learning Project

## Project Description

A machine learning project for analyzing and predicting energy consumption in buildings. The goal is to build a predictive model that forecasts energy consumption based on building characteristics and environmental conditions.

## Dataset

**Source**: [Kaggle - Energy Consumption Dataset](https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression)

### Data Structure:
- **Training data**: 1000 samples (indices 0-999)
- **Test data**: 100 samples (indices 1000-1099)
- **Target variable**: `Energy Consumption`

### Features:
- `Square Footage` - building area (ft²)
- `Number of Occupants` - number of residents/employees
- `Appliances Used` - number of appliances in use
- `Average Temperature` - average temperature
- `Building Type` - building type (0: Residential, 1: Commercial, 2: Industrial)
- `Is weekend` - whether it's weekend (0: weekdays, 1: weekend)

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

## Data Preprocessing

1. **Dataset merging**: Combining train and test data into one DataFrame
2. **Feature Engineering**: 
   - `Day of Week` → `Is weekend` (binarization)
   - `Building Type` → numerical encoding (0, 1, 2)
3. **Data splitting**: Maintaining original train/test split while preserving data consistency after type conversion

## Exploratory Data Analysis (EDA)

### Visualizations:
- **Pairplot**: Correlation matrix plots between all variables
- **Correlation heatmap**: Pearson correlation matrix
- **Box plots**: Energy consumption distribution by:
  - Building type (Residential/Commercial/Industrial)
  - Days of week (Weekday/Weekend)
- **Scatter plots**: Relationships with color coding by categorical variables

### Key Observations:
- Strong positive correlation between `Square Footage` and `Energy Consumption`
- Industrial buildings consume the most energy
- Differences in consumption between weekdays and weekends

## Machine Learning Models

### Compared Algorithms:
1. **Linear Regression (LR)**
   - Simple baseline model
   - Assumes linear relationships

2. **Random Forest (RF)**
   - Ensemble method
   - Better handling of categorical variables
   - Automatically detects interactions between variables

### Evaluation Metrics:
- **MSE** (Mean Squared Error)
- **R²** (R-squared) - coefficient of determination
- **MAE** (Mean Absolute Error)

## Results

Model comparison on test data:

| Model | MSE | R² | MAE |
|-------|-----|----|----- |
| Linear Regression | 0.00 | 1.0000 | 0.01 |
| Random Forest | 14276.00 | 0.9792 | 94.96 |

### Results Visualization:
- Scatter plots: actual values vs predictions
- Perfect line (y = x) for comparison
- R² score for each model

## Code Structure

```python
# 1. Library imports and data download
# 2. Preprocessing and feature engineering
# 3. Exploratory Data Analysis (EDA)
# 4. Train/test split
# 5. Model training
# 6. Evaluation and results comparison
# 7. Prediction visualization
```

## How to Run

1. Clone the repository
2. Install required libraries
3. Run code in Jupyter/IDE with `#%%` support (VS Code, PyCharm)
4. Code automatically downloads dataset from Kaggle

## Conclusions

- Linear Regression performed better than Random Forest:
  - MSE for Linear Regression equals 0, while Random Forest = 14276
  - R² equals 1 for Linear Regression, showing perfect fit
  - MAE close to 0 for Linear Regression

- Most important energy consumption predictors:
  - Building area (`Square Footage`)
  - Building type (`Building Type`)
  - Number of appliances (`Appliances Used`)

## Possible Improvements

1. **Feature Engineering**:
   - Variable interactions (e.g., Square_Footage × Building_Type)
   - Polynomial features
   - Normalization/standardization

2. **Additional Models**:
   - Gradient Boosting (XGBoost, LightGBM)
   - Support Vector Regression
   - Neural Networks

3. **Hyperparameter Tuning**:
   - Grid Search
   - Random Search
   - Bayesian Optimization

4. **Cross-Validation**:
   - K-fold CV on training data
   - Stratified sampling

## Author

Igor Bukowski

## License

MIT License