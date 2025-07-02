# label_coder.py

# Mapa dla 'workclass':
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

# Mapa dla 'education':
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

# Mapa dla 'marital-status':
marital_status_map = {
    'Never-married': 0,
    'Married-spouse-absent': 1,
    'Separated': 2,
    'Divorced': 3,
    'Widowed': 4,
    'Married-AF-spouse': 5,
    'Married-civ-spouse': 6
}

# Mapa dla 'occupation':
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

# Mapa dla 'relationship':
relationship_map = {
    'Own-child': 0,
    'Other-relative': 1,
    'Not-in-family': 2,
    'Unmarried': 3,
    'Wife': 4,
    'Husband': 5
}

# Mapa dla 'race':
race_map = {
    'Amer-Indian-Eskimo': 0,
    'Other': 1,
    'Asian-Pac-Islander': 2,
    'Black': 3,
    'White': 4
}

# Mapa dla 'gender':
gender_map = {
    'Female': 0,
    'Male': 1
}

# Mapa dla 'native-country':
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

# Jeden słownik grupujący wszystkie mapowania
data_coders = {
    'workclass': workclass_map,
    'education': education_map,
    'marital-status': marital_status_map,
    'occupation': occupation_map,
    'relationship': relationship_map,
    'race': race_map,
    'gender': gender_map,
    'native-country': native_country_map
}