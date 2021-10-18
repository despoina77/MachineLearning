import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# Heart Failure Detection
""")
image = Image.open('D:\pythonProject\ML\Heart_failure\heart.jpeg')
st.image(image, caption='ML_Heart', use_column_width=True)

df = pd.read_csv('D:\pythonProject\ML\Heart_failure\heart.csv')
df.head()


df_eight =  df[['Age','RestingBP','Cholesterol','FastingBS','MaxHR']]

st.subheader('Data')
st.dataframe(df_eight)
st.write(df_eight.head())

chart = st.line_chart(df_eight)

X = df_eight.iloc[:, 0:5].values
Y = df_eight.iloc[:, -1].values
#split to 75% training 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

def get_user_input():
    Age = st.sidebar.slider('Age',0, 77, 50)
    RestingBP = st.sidebar.slider('RestingBP', 0, 200, 80)
    Cholesterol = st.sidebar.slider('Cholesterol', 0, 603, 150)
    FastingBS = st.sidebar.slider('FastingBS', 0, True)
    MaxHR = st.sidebar.slider('MaxHR', 0, 202, 60)

    user_data = {'Age':Age,
                 'RestingBP':RestingBP,
                 'Cholesterol':Cholesterol,
                 'FastingBS':FastingBS,
                 'MaxHR':MaxHR}

    features = pd.DataFrame(user_data, index=[0])
    return features
user = get_user_input()

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

st.subheader('Model Score: ')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

prediction = RandomForestClassifier.predict(user)

#Classification
st.subheader('Classification')
st.write(prediction)



