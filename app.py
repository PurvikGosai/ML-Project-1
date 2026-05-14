## THIS FILE WILL BE OUR ACTUAL FRONTEND

import streamlit as st
import numpy as np
import pandas as pd
import pickle

## GUAGE CHART
import plotly.graph_objects as go
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

import os

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

with tab2:
    
with tab3:
    st.subheader("Download Your Report here...")
    def generate_pdf(prob_log,input_data):
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Startup prediction Report",styles["Title"]))
        content.append(Paragraph(f"Success Probability:{prob_log*100:.2f}%",styles["Normal"]))
        content.append(Paragraph("Import Details:",styles["Heading2"]))

        for col in input_data.columns:
            value = input_data.iloc[0][col]
            content.append(Paragraph(f"{col}:{value}",styles["Normal"]))

        doc.build(content)
        return "report.pdf"
    
    if st.button("Download Report"):
        pdf_file=generate_pdf(prob_log, input_data)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="Startup_Report.pdf",
                mime="application/pdf"
            )
    def save_data(input_data,prob_log):
        input_data["Prediction"] = prob_log
        file_path = "Data/Client_history.csv"


        if os.path.exists(file_path):
            input_data.to_csv(file_path, mode = 'a',header=False, index = False)
        else:
            input_data.to_csv(file_path ,index=False)
    if st.button("Save Data"):
        save_data(input_data.copy(), prob_log) ##timepass
        st.success("Data saved successfully!")

    if os.path.exists("Data/client_history.csv"):
        st.subheader("past prediction history")
        history = pd.read_csv("Data/client_history.csv")
        st.dataframe(history.tall(10))

    