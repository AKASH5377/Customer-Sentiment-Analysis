import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

reviews = [
    "Great product",
    "Amazing service",
    "Poor quality",
    "Bad experience",
    "Excellent support",
    "Not satisfied"
]

labels = [1, 1, 0, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

model = LogisticRegression()
model.fit(X, labels)

test_review = ["The product is amazing"]
test_vector = vectorizer.transform(test_review)

prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("Positive Review")
else:
    print("Negative Review")
