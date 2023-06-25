import pickle
import sys

from common import clean_tokenize

count_vectorizer_file = "./models/count_vectorizer.pickle"
model_file = "./models/spam_model.pickle"

cv = pickle.load(open(count_vectorizer_file, "rb"))
saved_model = pickle.load(open(model_file, "rb"))

def prediction(text):
    text = [clean_tokenize(text)]
    cv_text = cv.transform(text)
    prediction = saved_model.predict(cv_text)

    if prediction[0] == 1:
        return "Spam!"
    
    else:
        return "Not Spam!"








