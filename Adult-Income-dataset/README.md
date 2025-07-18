# Data Analysis: Adult Income Dataset (Census Income)

## 📋 Project Description

The goal of this project is to classify individuals' income based on demographic data collected by the US Census Bureau. The target variable is `income` — whether a person earns more than 50K annually (`>50K`) or not (`<=50K`).

The project uses the **Adult Income Dataset** available on [Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset), originally developed by Ronny Kohavi and Barry Becker.

---

## 🧠 Classification Methods Used

This project applies three classic machine learning algorithms:

- **Decision Tree (`DecisionTreeClassifier`)**
- **Naive Bayes (manual implementation)**
- **K-Nearest Neighbors (`KNN`)**

---

## 📈 Results

| Model             | Accuracy | `>50K` Class F1-score | Remarks                         |
|-------------------|----------|-----------------------|--------------------------------|
| KNN               | 0.7818   | 0.63                  | Good results, but worse than Bayes and Tree |
| Naive Bayes       | 0.8566   | 0.69                  | Highest accuracy, good precision |
| Decision Tree     | 0.8106   | 0.63                  | Balanced trade-off between precision and recall |

#### Sample Classification Report (Naive Bayes):

