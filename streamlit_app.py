import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pycaret
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pycaret.classification import *

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

st.subheader('accuracy with k-nn')

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.20, random_state = 0)
model = pickle.load(open('model.pkl', 'rb'))
y_pred = model.predict(x_test)
cv_scores = cross_val_score(model, x_train, y_train, cv=10)
accuracy_score = pd.DataFrame(data=[cv_scores.mean()], columns=['accuracy'])

st.write(accuracy_score)

st.subheader('compare wth other model')

s = setup(data = df_x_scaled, target = y, train_size=0.8, session_id=0)
modified_knn = create_model('knn', n_neighbors=2, metric='euclidean')
default_models = ['dt', 'rf', 'svm', 'nb', 'lr', 'gbc', 'ada', 'qda', 'et', 'xgboost', 'lightgbm', 'ridge', 'lda', 'dummy']

if st.button("compare with other model"):
    comparing = compare_models(include=[modified_knn] + default_models)
    results = pull()
    st.write(results)

