import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
import json
from flask import Flask, request, jsonify, render_template


path = "C:/Users/paul.bonte/Formation OC/P7_Bonte_Paul"
df = pd.read_csv((path + "/train.csv"))
valid_ids = df['SK_ID_CURR'].values.tolist()
cols_no_graph = df[['SK_ID_CURR', 'TARGET']]
df_columns = df.columns
df_small = df.head(100)



def filter_dataset():
    'Filters dataset down to a single line, being the chosen client ID'
    df_small = df[df['SK_ID_CURR'] == int(client_id)]
    df_small.drop(columns = ['SK_ID_CURR'], inplace = True)
    return df_small


def get_prediction():
    
        url = 'http://127.0.0.1:5000/predict'
        param = {'id_client' : client_id}
        # Post JSON file
        r = request.post(url, json = param)
        # Visualize response
        st.write(r)



st.title("Credit risk assesment dashboard")

client_id = st.text_input('Client ID:')
if client_id == '':
        st.write('Please enter a Client ID.')
else:
       if int(client_id) not in valid_ids:
            st.markdown(':exclamation: This ID does not exist. Please enter a valid one.')
       else:
            # Get prediction
            pred = get_prediction()
            st.bar_chart(pred)
           
select = st.selectbox('choose variable 1',df_columns)
feature_data = df_small[select]
st.bar_chart(feature_data)

select_2 = st.selectbox('choose variable 2',df_columns)
feature_data_2 = df_small[select_2]
st.bar_chart(feature_data_2)

select_3 = st.selectbox('choose variable X',df_columns)
feature_data_3 = df_small[select_3]
select_4 = st.selectbox('choose variable Y',df_columns)
feature_data_4 = df_small[select_4]
data_graph = pd.concat([feature_data_3 , feature_data_4] , axis = 1)

st.bar_chart(data_graph)

