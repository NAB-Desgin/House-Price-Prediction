import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_icon="🏠", page_title="House Price Prediction",layout="wide")
with open("Rf_model.joblib","rb")as file:
    model=joblib.load(file)

df=pd.read_csv("cleaned_df.csv")
with st.sidebar:
    st.title("House Price Prediction App")
    st.image("house_logo.png",width=250)
    st.image("house_logo2.png",width=250)
    st.image("house_logo3.png",width=250)
    st.image("house_logo4.png",width=250)

st.header("House Price Prediction")
with st.container(border=True):
    col1,col2=st.columns(2)

    with col1:
        st.image("house_logo.png",width=250)
        st.image("house_logo2.png",width=250)
        
    with col2:
        st.image("house_logo3.png",width=250)
        st.image("house_logo4.png",width=250)
        
  


with st.container(border=True):
    col1,col2=st.columns(2)

    with col1:
        location=st.selectbox("location : ",options=df['location'].unique())
        sqft=st.number_input("Sq.ft",min_value=300)
    with col2:
        bath=st.selectbox("bath : ",options=sorted(df['bath'].unique()))
        bhk=st.selectbox("bhk : ",options=sorted(df['bhk'].unique()))

def get_encoded_loc(location):
    for loc,encoded in zip(df['location'],df['encoded_loc']):
        if location==loc:
            return encoded
        
encoded_loc=get_encoded_loc(location)

if st.button("Predict"):
    inp_data=[[sqft,bath,bhk,encoded_loc]]
    pred=model.predict(inp_data)
    pred=float(f'{pred[0]:2f}')
    st.title(f"Predicted Price: Rs. {pred*100000}")