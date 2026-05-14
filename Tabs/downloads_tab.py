import streamlit as st
import pandas as pd
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

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


def show_downloads(prob_log,input_data):
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
