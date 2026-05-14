import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
def show_prediction(prob_log,log_model,input_values)


    st.subheader("Prediction Result")
    st.metric("success probability:",f"{prob_log*100:.2f}")

    if prob_log > 0.5:
        st.success("Stratup likely to succeed")
    else:
        st.error("high risk of failure  ")
        col1,col2=st.columns(2)

    ## --------------LEFT : PREDICTION 
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_log * 100,
            title={'text' : "Success Probability"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig)
        
    ## --------------RIGHT : PREDICTION
    with col2:
        st.subheader("Feature Importance")
        features = ["Experience","Team","Funding","Market","Innovation","Marketing","Competition","Industry","Education","Stage","Revenue"]
        importance = log_model.coef_[0]
        df_img = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })
        fig2 = px.bar(df_img, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2)
        