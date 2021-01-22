# IMPORT STATEMENTS
from typing import Any, Union

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

from numpy import loadtxt
from urllib.request import urlopen


@st.cache()
def Load_File():
    # path = r"C:\pima-indians-diabetes.csv"
    path = r'/Users/hughieholness/Desktop/Machine_Learning/datasets/pima-indians-diabetes.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    return pd.read_csv(path, names=names)

def Load_Url():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    raw_data = urlopen(url)
    return loadtxt(raw_data, delimiter=",")

dataset = Load_Url()
# st.write("url load", dataset.shape)
df = pd.DataFrame(dataset)
df.columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# st.write("DataFrame", df1.head())


# df = Load_File()


# HEADINGS
st.title('Diabetes Prediction ')
st.sidebar.header('Enter Patient Data')
st.header('using Machine Learnt Data')
st.subheader('Enter Patient Details using sliders on the left')
# st.write(df.describe())

# X AND Y DATA
x = df.drop(['class'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('(1/8): Num of Pregnancies', 0,17, 2 )
  glucose = st.sidebar.slider('(2/8): Plasma glucose concentration a 2 hours in an oral glucose tolerance test', 0,200, 120 )
  bp = st.sidebar.slider('(3/8): Diastolic blood pressure (mm Hg)', 0,122, 70 )
  skinthickness = st.sidebar.slider('(4/8): Triceps skin fold thickness (mm)', 0,100, 20 )
  insulin = st.sidebar.slider('(5/8): 2-Hour serum insulin (mu U/ml)', 0,846, 79 )
  bmi = st.sidebar.slider('(6/8): Body mass index (weight in kg/(height in m)^2)', 0,67, 20 )
  dpf = st.sidebar.slider('(7/8): Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('(8/8): Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data to Predict')
st.write(user_data)
trans = pd.DataFrame(df)

# machine learning MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

# Age vs Pregnancies
# st.write("Columns", df.columns)
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(data = df, x = 'age', y = 'preg',  hue = 'class', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'age', y = 'plas', data = df, hue = 'class' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'pres', data = df, hue = 'class', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'age', y = 'skin', data = df, hue = 'class', palette='Blues')
ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'age', y = 'plas', data = df, hue = 'class', palette='rocket')
ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'age', y = 'mass', data = df, hue = 'class', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'age', y = 'pedi', data = df, hue = 'class', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)

# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are probably not Diabetic'
else:
  output = 'You are probably Diabetic'
st.title(output)
st.subheader('Model Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')