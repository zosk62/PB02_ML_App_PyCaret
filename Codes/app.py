import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model
from pycaret.regression import setup, compare_models, pull, save_model, load_model, predict_model
import matplotlib.pyplot as plt


with st.sidebar:
  st.image('img01.jpg')
  st.title("SmartML Explorer")
  navigation_options = ["Upload Dataset", "Data Profiling", "Machine Learning", "Download Model", "Make Predictions"]
  choice = st.radio("Select Action", navigation_options)
  st.info("Explore your data effortlessly with ML_AutoStream! This application leverages the power of PyCaret, a versatile Python library, to simplify and enhance your machine learning journey. From automated data profiling to predictive modeling, ML_AutoStream uses PyCaret's capabilities to deliver accurate and efficient results.")

  
if os.path.exists("dataset.csv"):
  df2=pd.read_csv("dataset.csv",index_col=None )
  
  
st.markdown("<h4 style='text-align: center; color: #2E7D32;'>Streamlit App for Automated Machine Learning with PyCaret</h4>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #2E7D32;'>", unsafe_allow_html=True)

if choice == "Upload Dataset":
  st.title("Upload")
  file = st.file_uploader("Upload your dataset here")
  if file:
    df1 = pd.read_csv(file, index_col=None)
    df1.to_csv("dataset.csv", index = None)
    st.dataframe(df1)
    
if choice == "Data Profiling":
  st.title("Automated exploratory data analysis")
  profile_report = df2.profile_report()
  st_profile_report(profile_report)

if choice == "Machine Learning":
  st.title("Machine Learning - Exploration and Model Comparison")
  target= st.selectbox("Select yout target", df2.columns)
  with open('target.txt', 'w') as f:  # 타겟 변수 저장
      f.write(target)
  if st.button("train model"):
    setup(df2, target=target)
    setup_df=pull()
    st.info("Machine Learning Experiment Settings:")
    st.dataframe(setup_df)
    best_model = compare_models()
    compare_df=pull()
    st.info("Best Machine Learning Model:")
    st.dataframe(compare_df)
    st.info("The model below is the best-performing model.")
    best_model
    save_model(best_model, 'best_model')

if choice == "Download Model":
  st.title("Download the model")
  with open("best_model.pkl", 'rb') as f:
    st.download_button("download the model", f, "trained_model.pkl")

if choice == "Make Predictions":
  st.title("Make Predictions")
  model = load_model('best_model')
  with open('target.txt', 'r') as f:  # 타겟 변수 불러오기
    target = f.read().strip()
  inputs = {}
  for col in df2.columns:
    if df2[col].dtype == 'object':  # categorical variable
      values = df2[col].unique()
      inputs[col] = st.selectbox(f"Select value for {col}", options=values)
    else:  # numerical variable
      mean_value = df2[col].mean()
      inputs[col] = st.number_input(f"Enter value for {col}", value=mean_value)
  
  st.info("Select X-Axis and Y-Axis for plot:")
  x_col = st.selectbox('Select a column for the x axis', df2.columns)
  y_col = st.selectbox('Select a column for the y axis', [target])
  if st.button("Predict"):
    data = pd.DataFrame([inputs])
    st.dataframe(data)
    prediction = predict_model(model, data=data)
    st.info("The prediction is:")
    st.write(prediction[target])  

    

    plt.figure(figsize=(12, 8))
    plt.plot(df2[x_col], df2[y_col], 'ko-', label='Actual Value') 
    plt.scatter(data[x_col], prediction[y_col], color='r', label='Predicted Value')
    plt.plot(data[x_col], prediction[y_col], 'r--', linewidth=2)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Actual and Predicted Values')
    plt.legend()

    st.pyplot(plt)