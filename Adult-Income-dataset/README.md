# Analiza danych: Adult Income Dataset (Census Income)

## 📋 Opis projektu

Celem projektu jest klasyfikacja dochodów osób na podstawie danych demograficznych zebranych przez US Census Bureau. Przewidywana zmienna to `income` – czy dana osoba zarabia więcej niż 50K rocznie (`>50K`) czy nie (`<=50K`).

Projekt opiera się na zbiorze danych **Adult Income Dataset** dostępnym na [Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset), który został pierwotnie opracowany przez Ronny’ego Kohaviego i Barry’ego Beckera.

---

## 🧠 Wykorzystane metody klasyfikacji

W projekcie zastosowano trzy klasyczne algorytmy uczenia maszynowego:

- **Drzewo decyzyjne (`DecisionTreeClassifier`)**
- **Naive Bayes (ręczna implementacja modelu)**
- **K-Nearest Neighbors (`KNN`)**

---

## 📈 Wyniki

| Model               | Accuracy | Klasa `>50K` F1-score | Uwagi |
|---------------------|----------|------------------------|--------|
| KNN                 | 0.7818   | 0.63                   | Dobre wyniki, ale gorsze niż Bayes i Drzewo |
| Naive Bayes         | 0.8566   | 0.69                   | Najwyższa dokładność, dobra precyzja |
| Drzewo decyzyjne    | 0.8106   | 0.63                   | Poprawny kompromis precyzja/czułość |

#### Przykładowy raport klasyfikacji (Naive Bayes):

```
Dokładność: 0.8566
Raport klasyfikacji:
              precision    recall  f1-score   support

       <=50K       0.88      0.93      0.91     11285
        >50K       0.75      0.64      0.69      3775

    accuracy                           0.86     15060
   macro avg       0.82      0.78      0.80     15060
weighted avg       0.85      0.86      0.85     15060
```

---

## 🧼 Przetwarzanie danych

1. Wczytanie danych z Kaggle.
2. Usunięcie rekordów z brakującymi wartościami (`?`).
3. Przekształcenie zmiennych kategorycznych do wartości liczbowych za pomocą mapowania (`Label Encoding` z kontrolowaną kolejnością, np:klasa robotnicza przez stabliność i z preferencją publicznego sektora, status cywilny przez społeczno-ekonomiczną preferencje, kraj przez rozwój ekonomiczny).
4. Podział zbioru danych na dane treningowe (30162 rekordy) i testowe (15060 rekordów).
5. Konwersja wszystkich danych na `int32` dla optymalizacji pamięci.

---

## 🔧 Wykorzystane biblioteki

- `pandas`, `numpy`
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.metrics` – accuracy, classification report
- `kagglehub` – do pobierania zbioru danych
- Implementacja Bayesa i KNN była częściowo ręczna (Bayes – bez użycia `sklearn.naive_bayes`)

---

## 📂 Struktura kodu

- Przetwarzanie danych (`data cleaning`, encoding)
- Trening i predykcja: Drzewo Decyzyjne (z pruningiem), Naive Bayes (manualny), KNN
- Ewaluacja każdego modelu: `accuracy`, `classification_report`
- Mapowanie kategoryczne z zachowaniem sensownej kolejności społeczno-ekonomicznej

---

## 📊 Wnioski

- **Naive Bayes** uzyskał najlepszy wynik dokładności (ponad 85%).
- Dane są silnie niezbalansowane (`<=50K`: ~76%, `>50K`: ~24%).
- Modele różnie radzą sobie z klasą mniejszościową – precyzja dla `>50K` była najniższa w KNN.
- Przemyślane mapowanie kategoryczne (np. poziomy edukacji) poprawiło jakość predykcji.

---

## 📁 Źródła

- Dataset: [Adult Income Dataset – Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
- Źródło oryginalne: UCI Machine Learning Repository
- Referencja naukowa: Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.*

---

## ✍️ Autor

Projekt wykonany w ramach zadania klasyfikacji binarnej.  
Autor: igor bukowski
