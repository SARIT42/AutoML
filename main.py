import streamlit as st
import pandas as pd
import os
import sklearn
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup,pull,compare_models,save_model


with st.sidebar:
    st.image('https://www.marktechpost.com/wp-content/uploads/2022/07/Blog-Banner-4.png')
    st.title("Auto-ML Classification App")
    choice = st.radio("Navigation", ["Home", "Upload", "Profiling/EDA", "ML Classification", "Download"])
    st.info("This application helps you to build an automated ML Classification pipeline model using Streamlit, Pandas Profiling and PyCaret. Magic!")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

if choice == "Home":
    st.header('Welcome to:')
    st.title('Auto-ML Classification App')
    st.write('Get the best AUTOMATED ML Model and EDA for your dataset in just a few clicks!')
    st.write('1. Upload your classification dataset in the Upload section in sidebar.')
    st.write('2. Get you automated EDA report using Pandas Profiling in the Profiling/EDA section in sidebar.')
    st.write('3. Get your best ML classification model comparisons in the Ml Classification section in sidebar')
    st.write('4. Download the best model in .pkl format from the Download section in the sidebar.')

if choice == 'Upload':
    st.title('Upload the dataset for modelling:')
    file = st.file_uploader("Upload your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.write('Dataframe Preview:')
        st.dataframe(df)


if choice == 'Profiling/EDA':
    st.title('Automated Exploratory Data Analysis')
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    print(sklearn.__version__)

if choice == 'ML Classification':
    st.title('ML Model results and comparisons')
    target = st.selectbox("Select Target Column:",df.columns)
    if st.button("Train Model"):
        setup(df, target=target, silent=True,fold_shuffle=True,
              imputation_type='iterative')
        setup_df = pull()
        st.info('This is the ML experiment settings')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info('This is the ML model')
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')

if choice == 'Download':
    st.title('Download the Model')
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the Best Model", f, "trained_model.pkl")




