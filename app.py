## THIS FILE WILL BE OUR ACTUAL FRONTEND

import streamlit as st
import pandas as pd
import pickle

from Tabs.prediction_tab import show_prediction
from Tabs.analytic_tab import show_analytics
from Tabs.downloads_tab import show_downloads

# st.title("Startup Success Predictor APP -By Comact")

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center; white-space:nowrap;'>Startup Success Predictor App </h1>",unsafe_allow_html=True)

st.sidebar.header("Startup Parameters")

experience = st.sidebar.slider("Founder Experience", 0, 25, 5)
team = st.sidebar.slider("Team Size", 1, 60, 10)
funding = st.sidebar.slider("Funding (Million $)", 0.1, 50.0, 5.0)
market = st.sidebar.slider("Market Size", 1, 10, 5)
innovation = st.sidebar.slider("Innovation Score", 1, 10, 6)
marketing = st.sidebar.slider("Marketing Budget", 0.1, 28.0, 3.0)
competition = st.sidebar.slider("Competition Level", 1, 10, 5)
revenue = st.sidebar.slider("Revenue Growth", -10, 100, 20)

industry = 1
education = 1
stage = 2

input_data = pd.DataFrame([{
    "FounderExperience": experience,
    "TeamSize": team,
    "FundingAmount": funding,
    "MarketSize": market,
    "InnovationScore": innovation,
    "MarketingBudget": marketing,
    "CompetitionLevel": competition,
    "IndustryType": industry,
    "FounderEducation": education,
    "ProductStage": stage,
    "RevenueGrowth": revenue
}])
# print("Founder Experience :",input_data[0][0])
# print("Team SIze :",input_data[0][1])
# print("Funding :",input_data[0][2])
# print("market :",input_data[0][3])



log_model = pickle.load(open("Model/logistic_model.pkl","rb"))

prob_log = log_model.predict_proba(input_data)[0][1]
print("Team size ",input_data["TeamSize"][0])
    # print("Probability of success",prob_log)



## LAYOUT


tab1, tab2, tab3 = st.tabs(["Prediction","Analytics","Downloads"])

with tab1:
    show_prediction(prob_log,log_model,
        [experience, team , funding, market,
         innovation, marketing, competition])
    

with tab2:
    show_analytics(df, knn_model, input_data)


with tab3:
    show_downloads(prob_log,input_data)
    
    