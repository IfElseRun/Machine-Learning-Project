import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle

st.title('Iowa Ames House Price Prediction')

st.write("""

# Ames Housing Prices
------------
This app predicts Ames Housing Prices using a machine learning model powered by [Scikit Learn](https://scikit-learn.org/).
Play with the values via the sliders on the left panel to generate new predictions.
""")


st.write("---")

train_df = pd.read_csv('housing/df_final.csv')
test_df = pd.read_csv('housing/test_df_final.csv')

feature_list = list(train_df.columns)
feature_list.remove('SalePrice')

X = train_df[feature_list]
y = train_df['SalePrice']

st.sidebar.header('Specify Input Parameters - these will determine the predicted value.')

def features_from_user():
    house_age = st.sidebar.slider('House Age', int(train_df['House_Age'].min()), int(train_df['House_Age'].max()), int(train_df['House_Age'].mean()))
    gr_liv_area = st.sidebar.slider('Living Area', int(train_df['Gr Liv Area'].min()), int(train_df['Gr Liv Area'].max()), int(train_df['Gr Liv Area'].mean()))
    overall_qual = st.sidebar.slider('Overall Quality', int(train_df['Overall Qual'].min()), int(train_df['Overall Qual'].max()), int(train_df['Overall Qual'].mean()))
    totrms = st.sidebar.slider('Total room above grade', int(train_df['Mas Vnr Area'].min()), int(train_df['Mas Vnr Area'].max()), int(train_df['Mas Vnr Area'].mean()))
    fullbath = st.sidebar.slider('Full bathrooms above grade', int(train_df['Full Bath'].min()), int(train_df['Full Bath'].max()), int(train_df['Full Bath'].mean()))
    garage_age = st.sidebar.slider('Garage Age', int(train_df['Garage_Age'].min()), int(train_df['Garage_Age'].max()), int(train_df['Garage_Age'].mean()))
    garage_area = st.sidebar.slider('Garage Area', int(train_df['Garage Area'].min()), int(train_df['Garage Area'].max()), int(train_df['Garage Area'].mean()))

    
    data = {
            'House_Age': house_age,
            'Garage Area': garage_area,
            '1st Flr SF': train_df['1st Flr SF'].loc[0],
            'Overall Qual': overall_qual,
            'Full Bath': fullbath,
            'Garage_Age': garage_age,
            'Gr Liv Area': gr_liv_area,
            'Total Bsmt SF': train_df['Total Bsmt SF'].loc[0],
            'Garage Cars': train_df['Garage Cars'].loc[0],
            'Mas Vnr Area': train_df['Mas Vnr Area'].loc[0]}
    
    input_data = pd.DataFrame(data, index=[0])
    
    return input_data
    
    
df = features_from_user()
ss_list = ['House_Age','Garage Area','1st Flr SF','Overall Qual','Mas Vnr Area','Full Bath','Garage_Age','Gr Liv Area','Total Bsmt SF','Garage Cars']

ss = StandardScaler()
X_train_sc = ss.fit_transform(X[ss_list])
df1 = ss.transform(df[ss_list])

st.write('---')

# Load the saved model
loaded_model = pickle.load(open('ridge_model.sav', 'rb'))

# Apply Model to Make Prediction
prediction = int(loaded_model.predict(df1))
prediction_nice = f"{prediction:,d}"


st.header('Prediction of House Price in Ames:')
st.write('Based on your selections, the model predicts a value of $%s.'%prediction_nice)
st.write('---')
            
            

