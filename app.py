# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:28:37 2022

@author: Paul.BONTE
"""

# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pandas as pd
import json
import lime
from lime import lime_tabular
from flask import Flask, request, jsonify


app = Flask(__name__)

path = "C:/Users/paul.bonte/Formation OC/P7_Bonte_Paul"
path2 = "C:/Users/paul.bonte/Formation OC/P7_Bonte_Paul/P7_Paul_Bonte"
data = pd.read_csv(path + "/data.csv")
model = pickle.load(open(path2 + "/model_credit.pkl","rb"))
exp = data.drop(columns = ["SK_ID_CURR"])

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(exp),
    feature_names= exp.columns,
    class_names=['0', '1'],
    mode='classification')

def filter_dataset(df, client_id):
    'Filters dataset down to a single line, being the chosen client ID'
    X = df[df['SK_ID_CURR'] == int(client_id)]
    X.drop(columns = ['SK_ID_CURR'], inplace = True)
    return X


def get_prediction(model , X):
    resultat = model.predict_proba(X)[0]
    return resultat

@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"text":"Hello, the API is up and running..." })


@app.route('/predict', methods=['POST'])
def predict():
    
    input = request.get_json(force= True)
    id_client = input["id_client"]
    X_id =filter_dataset(data , id_client)
    prediction = get_prediction(model , X_id)

    # Return output
    return jsonify({'Proba non défaut' : [float(round(prediction[0]*100,2)),int(50)] , 'Proba défaut' : [float(round(prediction[1]*100,2)),int(50)]})

@app.route('/lime' , methods=['POST'])
def explain() : 
    
    input = request.get_json(force= True)
    id_client = input["id_client"]
    explained = explainer.explain_instance(
    data_row=filter_dataset(data , id_client).iloc[0], 
    predict_fn=model.predict_proba)
    explained = explained.as_list()
    explained = pd.DataFrame(explained)
    explained[0] = explained[0].str.replace('\<= 0.00', '')
    explained[0] = explained[0].str.replace('\<=', '')
    explained[0] = explained[0].str.replace('\>', '')
    explained[0] = explained[0].str.replace('\<', '')
    explained[0] = explained[0].str.replace('\>=', '')
    #json_data = explained.to_json(orient='values')
    explained = pd.DataFrame.to_dict(explained)
      
    return jsonify(explained)
    
    
    

if __name__ == "__main__":
    app.run()