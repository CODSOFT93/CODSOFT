import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Example data (replace this with your actual data)
data = {
    'plot': [
        'A young man joins a rebellion to save the galaxy from an evil empire.',
        'A woman falls in love with a mysterious vampire.',
        'A detective solves a complex murder mystery in a rainy city.',
        'A team of superheroes come together to fight a cosmic threat.',
        'A comedian struggles with mental health issues and turns to crime.'
    ],
    'genre': ['Sci-Fi', 'Romance', 'Thriller', 'Action', 'Drama']
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['plot'], df['genre'], test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Choose a classifier (Logistic Regression here)
model = LogisticRegression(max_iter=1000)
# For Naive Bayes, use:
# model = MultinomialNB()

# Train
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(metrics.classification_report(y_test, y_pred))

# Test on new text
new_plot = ["A soldier travels through time to stop an alien invasion."]
new_tfidf = vectorizer.transform(new_plot)
predicted_genre = model.predict(new_tfidf)
print(f'Predicted genre for new plot: {predicted_genre[0]}')
