# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels
data["label_num"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Prepare X and y
X = data["message"]
y = data["label_num"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 3: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 5: Test the model
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 6: Test new messages
sample = [
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "Are we still on for the meeting tomorrow?",
    "URGENT! Your account has been suspended. Log in to verify.",
    "Congratulation you have won a cash prize"
]

sample_transformed = vectorizer.transform(sample)

predictions = model.predict(sample_transformed)

for msg, label in zip(sample, predictions):
    print(
        f"Message: {msg} → "
        f"{'Spam' if label == 1 else 'Ham'}"
    )

# Function that our website will use
def predict_spam(message):
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)[0]

    if prediction == 1:
        return "Spam"
    else:
        return "Ham"
