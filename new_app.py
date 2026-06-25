import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("Rf_model.joblib","rb") as file:
        return joblib.load(file)


model = load_model()


@st.cache_data
def load_data():
    return pd.read_csv("cleaned_df.csv")


df = load_data()

st.markdown("""
<style>


.main{
    background-color:#f7f9fc;
}


[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #0f2027,
        #203a43,
        #2c5364
    );
}

[data-testid="stSidebar"] h1{
    color:white;
    text-align:center;
}

.title{
    font-size:45px;
    font-weight:800;
    color:#203a43;
}

.subtitle{
    font-size:20px;
    color:#555;
}

.card{

    background:#1A313A;
    padding:25px;
    border-radius:20px;
    box-shadow:
    0px 5px 20px rgba(0,0,0,0.08);

}

.stButton>button{

    width:100%;
    height:55px;

    background:
    linear-gradient(
        90deg,
        #11998e,
        #38ef7d
    );

    color:black;
    font-size:22px;
    font-weight:bold;

    border-radius:15px;

}

.result{

    background:
    linear-gradient(
        135deg,
        #667eea,
        #764ba2
    );

    color:white;
    padding:30px;
    border-radius:25px;
    text-align:center;
    font-size:30px;
    font-weight:bold;

}

</style>
""",unsafe_allow_html=True)
with st.sidebar:
    st.title("🏠 Real Estate AI")
    st.image(
        "house_logo.png",
        width=250
    )

    st.markdown(
    """
    ### About App
    AI powered house price prediction system.
    Model:
    - Random Forest Regression
    - Location Encoding
    - Feature Based Prediction
    """
    )

st.markdown(
"""
<div class="title">
🏡 House Price Prediction
</div>

<div class="subtitle">
AI based property valuation system using Machine Learning
</div>

""",
unsafe_allow_html=True
)

st.write("")
col1,col2,col3 = st.columns(3)
with col1:
    st.image( "house_logo2.png",width=250)
with col2:
    st.image(
        "house_logo3.png",
        width=250
    )
with col3:
    st.image(
        "house_logo4.png",
        width=250
    )
st.write("")
st.markdown(
"""
<div class="card">
<h2>
📌 Enter Property Details
</h2>
</div>
""",
unsafe_allow_html=True)
col1,col2,col3,col4 = st.columns(4)
with col1:
    location = st.selectbox(
        "📍 Location",
        options=df['location'].unique())


with col2:
    sqft = st.number_input(
        "📐 Area (Sq.ft)",
        min_value=300,
        value=1000
    )


with col3:
    bath = st.selectbox(
        "🚿 Bathrooms",
        options=sorted(df['bath'].unique())
    )


with col4:
    bhk = st.selectbox(
        "🛏 BHK",
        options=sorted(df['bhk'].unique())
    )

def get_encoded_loc(location):
    encoded = df.loc[
        df['location']==location,
        'encoded_loc'
    ]

    return encoded.iloc[0]



encoded_loc = get_encoded_loc(location)



st.write("")

if st.button("🔮 Predict House Price"):
    input_data=[
        [
            sqft,
            bath,
            bhk,
            encoded_loc
        ]
    ]


    prediction=model.predict(input_data)


    price = prediction[0]*100000


    st.markdown(
    f"""

    <div class="result">

    🏠 Estimated Property Price

    <br><br>

    ₹ {price:,.0f}

    </div>

    """,
    unsafe_allow_html=True
    )

st.write("")

st.caption(
"Built using Machine Learning | Random Forest Regression | Streamlit"
)