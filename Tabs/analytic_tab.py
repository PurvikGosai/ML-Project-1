
def show_analytics(df)
st.subheader("Similar Startup")
    knn_model = pickle.load(open("Model/knn_model.pkl", "rb"))
    df = pd.read_csv("Data/startup_dataset.csv")
    distances, indices = knn_model.kneighbors(input_data)
    similar = df.iloc[indices[0]]
    similar["Distance"] = distances[0]
    st.dataframe(similar)
