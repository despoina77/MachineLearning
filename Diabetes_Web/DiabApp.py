import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Title
st.write("""
Diabetes Detection
""")

image = Image.open('D:\pythonProject\ML\Diabetes_Web\diabetes.jpg')
st.image(image, caption='ML_Diabetes', use_column_width=True)

#Data
df = pd.read_csv('diabetes.csv')

#set a subheader
st.subheader('Data')
st.dataframe(df)
st.write(df.describe())

chart = st.bar_chart(df)
#split data
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values
#split to 75% training 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#input from user
def user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    bloodPressure = st.sidebar.slider('bloodPressure', 0, 122, 72)
    skinThickness = st.sidebar.slider('skinThickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0, 846, 30)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    user_data = {'pregnancies':pregnancies,
                 'glucose':glucose,
                 'bloodPressure':bloodPressure,
                 'skinThickness':skinThickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
                 'age':age}
    #Transfor to dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

#store input into variable
user= user_input()

#create and train model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Model metrics
st.subheader('Model Score: ')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

#store prediction
prediction = RandomForestClassifier.predict(user)

#Classification
st.subheader('Classification')
st.write(prediction)

