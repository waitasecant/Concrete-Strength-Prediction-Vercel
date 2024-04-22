from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("RFModel.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    Cement = request.form.get("Cement")
    Blast_Furnace_Slag = request.form.get("Blast_Furnace_Slag")
    Fly_Ash = request.form.get("Fly_Ash")
    Superplasticizer = request.form.get("Superplasticizer")
    Age = request.form.get("Age")
    prediction = model.predict(
        pd.DataFrame(
            columns=[
                "Cement",
                "Blast_Furnace_Slag",
                "Fly_Ash",
                "Superplasticizer",
                "Age",
            ],
            data=np.array(
                [Cement, Blast_Furnace_Slag, Fly_Ash, Superplasticizer, Age]
            ).reshape(1, 5),
        )
    )
    return (
        str(np.round(prediction[0], 3)) + " " + "\N{PLUS-MINUS SIGN}" + " " + str(5.825)
    )


if __name__ == "__main__":
    app.run(debug=True)