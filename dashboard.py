import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib as plt
import matplotlib.pyplot as plt
#from flask import Flask, request, jsonify, render_template
import requests


path = "C:/Users/paul.bonte/Formation OC/P7_Bonte_Paul"
df = pd.read_csv((path + "/train.csv"))
valid_ids = df['SK_ID_CURR'].values.tolist()
cols_no_graph = df[['SK_ID_CURR', 'TARGET']]
df_columns = df.columns
df_columns_num = df.select_dtypes(exclude=["bool_","object_"]).columns
df_columns_cat = df.select_dtypes(exclude="number").columns


def example_ids():
    sample = df['SK_ID_CURR'].head(100000).sample(5).tolist()
    for i in range(0, len(sample)): 
        sample[i] = int(sample[i])
    st.write("Exemples d'ID clients :")
    st.write(str(sample).replace('[','').replace(']', ''))

def filter_dataset():
    'Filters dataset down to a single line, being the chosen client ID'
    df_small = df[df['SK_ID_CURR'] == int(client_id)]
    df_small.drop(columns = ['SK_ID_CURR'], inplace = True)
    return df_small


def get_prediction(client_id):
    
        url = 'http://127.0.0.1:5000/predict'
        param = {'id_client' : client_id}
        # Post JSON file
        r = requests.post(url, json = param)
        # Visualize response
        #df_pred = pd.json_normalize(r.json())
        df_pred = pd.DataFrame.from_dict(r.json(), orient='index').transpose()
        ax = df_pred.iloc[1].plot(kind ='line' , color = 'red')
        df_pred[['Proba non défaut','Proba défaut']].iloc[0].plot(kind='bar', ax=ax)
        return st.pyplot(ax.figure)
        
def get_explainer(client_id):
    
        url = 'http://127.0.0.1:5000/lime'
        param = {'id_client' : client_id}
        # Post JSON file
        r = requests.post(url, json = param)
        # Visualize response
        df_exp = pd.DataFrame.from_dict(r.json(), orient='columns')
        df_exp.set_index(list(df_exp)[0], inplace = True)
        ax = df_exp.plot(kind ='barh')

        return st.pyplot(ax.figure , clear_figure = True)


st.title("Evaluation du risque client")

client_id = st.text_input('Client ID:')
if client_id == '':
        st.write('Entre un ID client')
        example_ids()
        
else:
       if int(client_id) not in valid_ids:
            st.markdown(':exclamation: Cet identifiant n\'existe pas, veuillez en entrer un valide.')
       else:
            # Get prediction
            pred = get_prediction(client_id)
            exp = get_explainer(client_id)
            
            client_index = df[df['SK_ID_CURR']== int(client_id)].index.tolist()
            
            
            select = st.selectbox('Choisissez une variable numérique',df_columns_num.drop(cols_no_graph))
            feature_data = df[select]
            aa = df[select].iloc[client_index]
            ax = feature_data.plot(kind ='hist')
            st.write('La valeur pour le client est :', aa.iloc[0])
            st.pyplot(ax.figure, clear_figure = True)

            select_2 = st.selectbox('Choisissez une variable catégorielle',df_columns_cat)
            feature_data_2 = df[select_2]
            df_cat = feature_data_2.value_counts()
            ax = df_cat.plot(kind="pie")
            bb = df[select_2].iloc[client_index]
            st.write('La valeur pour le client est :', bb.iloc[0])
            st.pyplot(ax.figure, clear_figure = True)

            select_3 = st.selectbox('Choisissez une variable X',df_columns.drop(cols_no_graph))
            feature_data_3 = df[select_3]
            cc = df[select_3].iloc[client_index]
            select_4 = st.selectbox('Choisissez une variable Y',df_columns.drop(cols_no_graph))
            feature_data_4 = df[select_4]
            dd = df[select_4].iloc[client_index]
            x = np.array(feature_data_3)
            y = np.array(feature_data_4)
            ax = plt.scatter(x,y)
            #ax = data_graph.plot(kind = 'scatter', x = data_graph['AMT_CREDIT'] , y = data_graph['CNT_CHILDREN'])
            #st.dataframe(data_graph)
            st.write('La valeur X pour le client est :', cc.iloc[0])
            st.write('La valeur Y pour le client est :', dd.iloc[0])
            plt.scatter(cc.iloc[0] ,dd.iloc[0] , color ='red')
            st.pyplot(ax.figure)

