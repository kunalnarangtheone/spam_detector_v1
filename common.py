import nltk
import string
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stopwords_set = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def clean_token(token):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(token.lower()) if token.isalpha() \
    and token not in stopwords_set and token not in punctuations else ""

def clean_tokenize(text):
    tokens = word_tokenize(text)
    clean_tokens = [clean_token(token) for token in tokens]
    clean_tokens = [token for token in clean_tokens if token]
    return " ".join(clean_tokens)