import datetime, time

from flask import *
from predict import prediction

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def welcome():
    return "Welcome! To do spam predictions, call the /predict/ endpoint with the 'text' argument"

@app.route("/predict/", methods = ["GET"])
def predict():
    text = str(request.args.get("text"))
    return f"""
    Current time: {datetime.datetime.fromtimestamp(time.time()).strftime("%m-%d-%Y, %H:%M:%S")},\n
    Prediction: {prediction(text)}
    """

# sample_text = "Thank you for paying last month's bill. We're rewarding our very best customers with a gift for their loyalty. Click here!"

if __name__ == "__main__":
    app.run(port=7777)

