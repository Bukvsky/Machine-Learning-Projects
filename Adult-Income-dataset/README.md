Data Analysis: Adult Income Dataset (Census Income)
📋 Project Description
The goal of this project is to classify individuals' income based on demographic data collected by the US Census Bureau. The target variable is income — whether a person earns more than 50K annually (>50K) or not (<=50K).

The project uses the Adult Income Dataset available on Kaggle, originally developed by Ronny Kohavi and Barry Becker.

🧠 Classification Methods Used
This project applies three classic machine learning algorithms:

Decision Tree (DecisionTreeClassifier)

Naive Bayes (manual implementation)

K-Nearest Neighbors (KNN)

📈 Results
Model	Accuracy	>50K Class F1-score	Remarks
KNN	0.7818	0.63	Good results, but worse than Bayes and Tree
Naive Bayes	0.8566	0.69	Highest accuracy, good precision
Decision Tree	0.8106	0.63	Balanced trade-off between precision and recall

Sample Classification Report (Naive Bayes):
markdown
Kopiuj
Edytuj
Accuracy: 0.8566
Classification Report:
              precision    recall  f1-score   support

       <=50K       0.88      0.93      0.91     11285
        >50K       0.75      0.64      0.69      3775

    accuracy                           0.86     15060
   macro avg       0.82      0.78      0.80     15060
weighted avg       0.85      0.86      0.85     15060
🧼 Data Processing
Loading data from Kaggle.

Removing records with missing values (?).

Encoding categorical variables into numeric values using mapping (Label Encoding with controlled order, e.g., working class based on stability and public sector preference, marital status based on socio-economic preference, country based on economic development).

Splitting the dataset into training data (30,162 records) and test data (15,060 records).

Converting all data to int32 to optimize memory usage.

🔧 Libraries Used
pandas, numpy

sklearn.tree.DecisionTreeClassifier

sklearn.metrics – accuracy, classification report

kagglehub – for dataset downloading

Bayes and KNN implementations were partially manual (Bayes without sklearn.naive_bayes)

📂 Code Structure
Data processing (cleaning, encoding)

Training and prediction: Decision Tree (with pruning), Naive Bayes (manual), KNN

Evaluation of each model: accuracy, classification_report

Categorical mapping maintaining sensible socio-economic order

📊 Conclusions
Naive Bayes achieved the highest accuracy (over 85%).

The data is strongly imbalanced (<=50K: ~76%, >50K: ~24%).

Models vary in handling the minority class — precision for >50K was lowest with KNN.

Thoughtful categorical mapping (e.g., education levels) improved prediction quality.

📁 Sources
Dataset: Adult Income Dataset – Kaggle

Original source: UCI Machine Learning Repository

Scientific reference: Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.

✍️ Author
Project completed as a binary classification task.
Author: Igor Bukowski
