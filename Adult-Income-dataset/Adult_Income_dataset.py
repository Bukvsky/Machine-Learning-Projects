import kagglehub
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
# Download latest version
path = kagglehub.dataset_download("wenruliu/adult-income-dataset")


#removing unknown values
df = (pd.read_csv(path+"\\adult.csv",na_values="?",skipinitialspace=True)
        .dropna()
        )

for col in df.select_dtypes(include='object'):
    df[col] = df[col].str.strip()

# 1. workclass: Ordered by employment stability and public sector preference
workclass_map = {
    '?': 0,
    'Without-pay': 1,
    'Never-worked': 2,
    'Self-emp-not-inc': 3,
    'Self-emp-inc': 4,
    'Private': 5,
    'State-gov': 6,
    'Local-gov': 7,
    'Federal-gov': 8
}

# 2. education: Ordered by educational level (from elementary to postgraduate)
education_map = {
    'Preschool': 0,
    '1st-4th': 1,
    '5th-6th': 2,
    '7th-8th': 3,
    '9th': 4,
    '10th': 5,
    '11th': 6,
    '12th': 7,
    'HS-grad': 8,
    'Some-college': 9,
    'Assoc-voc': 10,
    'Assoc-acdm': 11,
    'Bachelors': 12,
    'Masters': 13,
    'Prof-school': 14,
    'Doctorate': 15
}

# 3. marital-status: Ordered by potential socioeconomic stability
marital_status_map = {
    'Never-married': 0,
    'Married-spouse-absent': 1,
    'Separated': 2,
    'Divorced': 3,
    'Widowed': 4,
    'Married-AF-spouse': 5,
    'Married-civ-spouse': 6
}

# 4. occupation: Based on skill level, specialization, and income trends (approximate)
occupation_map = {
    '?': 0,
    'Priv-house-serv': 1,
    'Handlers-cleaners': 2,
    'Other-service': 3,
    'Farming-fishing': 4,
    'Machine-op-inspct': 5,
    'Transport-moving': 6,
    'Craft-repair': 7,
    'Adm-clerical': 8,
    'Sales': 9,
    'Tech-support': 10,
    'Protective-serv': 11,
    'Exec-managerial': 12,
    'Prof-specialty': 13,
    'Armed-Forces': 14
}

# 5. relationship: Ordered by likely household responsibility (somewhat subjective)
relationship_map = {
    'Own-child': 0,
    'Other-relative': 1,
    'Not-in-family': 2,
    'Unmarried': 3,
    'Wife': 4,
    'Husband': 5
}

# 6. race: No natural order — included here only symbolically (use with caution)
race_map = {
    'Amer-Indian-Eskimo': 0,
    'Other': 1,
    'Asian-Pac-Islander': 2,
    'Black': 3,
    'White': 4
}

# 7. gender: Binary, Female = 0, Male = 1
gender_map = {
    'Female': 0,
    'Male': 1
}

# 8. native-country: Ordered roughly by economic development or immigration relevance
native_country_map = {
    '?': 0,
    'Honduras': 1,
    'Guatemala': 2,
    'Nicaragua': 3,
    'El-Salvador': 4,
    'Haiti': 5,
    'Dominican-Republic': 6,
    'Trinadad&Tobago': 7,
    'Columbia': 8,
    'Ecuador': 9,
    'Peru': 10,
    'Mexico': 11,
    'Cuba': 12,
    'Jamaica': 13,
    'Outlying-US(Guam-USVI-etc)': 14,
    'Cambodia': 15,
    'Laos': 16,
    'Vietnam': 17,
    'Thailand': 18,
    'Philippines': 19,
    'South': 20,
    'Hong': 21,
    'China': 22,
    'India': 23,
    'Iran': 24,
    'Japan': 25,
    'Taiwan': 26,
    'Poland': 27,
    'Portugal': 28,
    'Hungary': 29,
    'Greece': 30,
    'Yugoslavia': 31,
    'Italy': 32,
    'Ireland': 33,
    'Scotland': 34,
    'Germany': 35,
    'France': 36,
    'England': 37,
    'Holand-Netherlands': 38,
    'Canada': 39,
    'United-States': 40
}

df['native-country'] = df['native-country'].apply(lambda x: native_country_map.get(x, 0))
df['workclass'] = df['workclass'].map(workclass_map)
df['education'] = df['education'].map(education_map)
df['marital-status'] = df['marital-status'].map(marital_status_map)
df['occupation'] = df['occupation'].map(occupation_map)
df['relationship'] = df['relationship'].map(relationship_map)
df['race'] = df['race'].map(race_map)
df['gender'] = df['gender'].map(gender_map)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

df_train = df.iloc[:30162]
df_test = df.iloc[30162:]

# vectors
X_train = df_train.drop(columns='income').to_numpy()
y_train = df_train['income'].to_numpy()
X_test = df_test.drop(columns='income').to_numpy()
y_test = df_test['income'].to_numpy()


def knn(record, X_train, y_train, k=10):
    distances = np.linalg.norm(X_train-record,axis=1)
    nearest = np.argsort(distances)[:k]
    return Counter(y_train[nearest]).most_common(1)[0][0]

def main():
    predictions = []
    for i in range(len(X_test)):
        pred = knn(X_test[i], X_train, y_train, k=10)
        predictions.append(pred)
        if i % 1000 == 0:
            print(f"Processed {i} records...")

    acc = accuracy_score(y_test, predictions)
    print(f"\n✅ Accuracy: {acc:.4f}")



if __name__ == '__main__':
    main()