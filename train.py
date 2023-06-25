import pandas as pd
import pickle

from common import clean_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

df = pd.read_csv("./data/spam_dataset.csv", names =['is_spam', 'text'])

def clean_target(target):
    return 1 if target == "spam" else 0

df['is_spam'] = df['is_spam'].apply(clean_target)
df['text'] = df['text'].apply(clean_tokenize)

cv = CountVectorizer(strip_accents='unicode')
cv.fit(df['text'])
pickle.dump(cv, open("./models/count_vectorizer.pickle", "wb"))
X = cv.fit_transform(df['text'])

X_train, X_test, y_train, y_test = train_test_split(X, df['is_spam'], test_size=0.2, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy = {accuracy}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F1 = {f1}")
print(f"AUC = {auc_score}")

file_name = ("./models/spam_model.pickle")
pickle.dump(lr, open(file_name, 'wb'))

# saved_model = pickle.load(open(file_name, 'rb'))
# saved_model

