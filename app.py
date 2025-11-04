from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# Load the model safely using absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "iris_model.pickle")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def app_home():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        # Get form values and convert to float
        sepaL = float(request.form.get("sepaL"))
        sepaW = float(request.form.get("sepaW"))
        petL = float(request.form.get("petL"))
        petW = float(request.form.get("petW"))

        prediction_input = np.array([[sepaL, sepaW, petL, petW]])
        prediction_output = int(model.predict(prediction_input)[0])
       

        # Map prediction to flower name
        if prediction_output == 0:
            result = "Iris-setosa"
        elif prediction_output == 1:
            result = "Iris-versicolor"
        else:
            result = "Iris-virginica"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
