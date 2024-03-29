import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #la collecte des valeurs d'entree
    float_features = [float(x) for x in request.form.values()]
    dicte = {
        'Statut': 0,
        'Année': 0,
        'Polio': 0,
        'Diphtérie': 0,
        'VIH/SIDA': 0,
        'PIB': 0,
        'IMC': 0,
        'Scolarité': 0,
        'Alcool': 0,
        'Maigreur_1_19_ans': 0
    }
    i = 0
    for j in dicte.keys():
        if j == 'Statut' :
            continue
        dicte[j] = [float_features[i]]
        i += 1
    final_features = pd.DataFrame(dicte)

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Votre Espérence de vie est : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
