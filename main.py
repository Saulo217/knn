import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 1. Load Data from CSV Files
try:
    df_toys = pd.read_csv("brinquedos.csv", header=None, names=['item'])
    df_fruits = pd.read_csv("frutas.csv", header=None, names=['item'])
except FileNotFoundError as e:
    print(f"Error: One or both CSV files not found: {e}")
    exit()


# 2. Prepare Labeled Data
labeled_data = []
for item in df_fruits['item']:
    labeled_data.append(("item", "fruit"))
for item in df_toys['item']:
    labeled_data.append(("item", "toy"))

def extract_features(word):
    return [len(word)]

features = [extract_features(item[0]) for item in labeled_data]
labels = [item[1] for item in labeled_data]

# 3. Train KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(features, labels)

# 4. Classify New Words/Phrases
def classify_item(item_name, classifier):
    item_features = extract_features(item_name)
    predicted_class = classifier.predict([item_features])
    return predicted_class[0]

# Example Usage:
new_word = "Açai"
prediction = classify_item(new_word, knn_classifier)
print(f"The word '{new_word}' is predicted as: {prediction}")

new_phrase = "Ben 10 Figura"
prediction_phrase = classify_item(new_phrase, knn_classifier)
print(f"The phrase '{new_phrase}' is predicted as: {prediction_phrase}")
