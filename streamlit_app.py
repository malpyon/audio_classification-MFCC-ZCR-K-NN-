import streamlit as st
import pandas as pd
import pickle

st.write("""
# MFCC & ZCR audio classification with K-NN
""")

st.subheader('feature extraction dataset')

datasetku = pd.read_csv('audio_klas2.csv')
st.write(datasetku)

st.subheader('scaled dataset')

x = datasetku.iloc[:, 2:-2].values
y = datasetku.iloc[:, -1].values

scaler = pickle.load(open('scaler.pkl', 'rb'))
x_scaled = scaler.transform(x)
df_x_scaled = pd.DataFrame(data=x_scaled, columns=['MFCC'+str(x) for x in range(1,21)])
st.write(df_x_scaled)