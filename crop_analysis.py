import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("PENAMBANGAN DATA")
st.write("By: Salmatul Farida")
st.write("Grade: Penambangan Data C")
upload_data,deskripsi, preporcessing, modeling, implementation = st.tabs(["Upload Data",'deskripsi' ,"Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah crop analysis and prediction dataset yang diambil dari https://www.kaggle.com/code/theeyeschico/crop-analysis-and-prediction/data")
    st.write("Pada Dataset yang saya gunakan terdapat 7 fitur yaitu fitur N (Yang mewakili dari level kadar Nitrogen), fitur P (Mewakili kadar level Phosphorus), fitur K (Untuk level Pottassium yanga ada), fitur Temperature (Untuk temperatur dari tanahnya), fitur Humidity (dimana menunjukkan kelembabannya), fitur PH (menunjukkan PH dari tanah),rainfall (menampilkan data cirah hujan yang terjadi)")

    
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with deskripsi:

    st.write("Dataset yang digunakan adalah crop analysis and prediction dataset yang diambil dari https://www.kaggle.com/code/theeyeschico/crop-analysis-and-prediction/data")
    st.write("Pada Dataset yang saya gunakan terdapat 7 fitur yaitu fitur N (Yang mewakili dari level kadar Nitrogen), fitur P (Mewakili kadar level Phosphorus), fitur K (Untuk level Pottassium yanga ada), fitur Temperature (Untuk temperatur dari tanahnya), fitur Humidity (dimana menunjukkan kelembabannya), fitur PH (menunjukkan PH dari tanah),rainfall (menampilkan data cirah hujan yang terjadi)")
with preporcessing:
    st.write("""# Preprocessing""")
    df[['N','P','K','temperature','humidity','ph','rainfall','label']].agg(['min','max'])
    X = df.drop(labels = 'label',axis = 1)
    y = df['label']
    "### Normalize data hasil"
    X

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape


    labels = pd.get_dummies(df.label).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y

    le.inverse_transform(y)

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

# with modeling:

#     st.markdown("# Model")
#     # membagi data menjadi data testing(20%) dan training(80%)
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#     # X_train.shape, X_test.shape, y_train.shape, y_test.shape

#     nb = st.checkbox("Metode Naive Bayes")
#     knn = st.checkbox("Metode KNN")
#     dt = st.checkbox("Metode Decision Tree")
#     sb = st.button("submit")

#     #Naive Bayes
#     # Feature Scaling to bring the variable in a single scale
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     GaussianNB(priors=None)
#     # Fitting Naive Bayes Classification to the Training set with linear kernel
#     nvklasifikasi = GaussianNB()
#     nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

#     # Predicting the Test set results
#     y_pred = nvklasifikasi.predict(X_test)
        
#     y_compare = np.vstack((y_test,y_pred)).T
#     nvklasifikasi.predict_proba(X_test)

#     akurasi = round(100 * accuracy_score(y_test, y_pred))

#     #Decision tree
#     dt = DecisionTreeClassifier()
#     dt.fit(X_train, y_train)

#     # prediction
#     dt.score(X_test, y_test)
#     y_pred = dt.predict(X_test)
#     #Accuracy
#     akur = round(100 * accuracy_score(y_test,y_pred))

#     K=10
#     knn=KNeighborsClassifier(n_neighbors=K)
#     knn.fit(X_train,y_train)
#     y_pred=knn.predict(X_test)

#     skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    

#     if nb:
#         if sb:

#             """## Naive Bayes"""
            
#             st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))

#     if knn:
#         if sb:
#             """## KNN"""

#             st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    
#     if dt:
#         if sb:
#             """## Decision Tree"""
#             st.write('Model Decission Tree Accuracy Score: {0:0.2f}'.format(akur))

with implementation:
    st.write("# Implementation")
    Weight = st.number_input('Masukkan Berat Mangga')
    Length = st.number_input('Masukkan Panjnag Mangga')
    circumference = st.number_input('Masukkan Lingkar Mangga')

    def submit():
        # input
        inputs = np.array([[1,
            Weight,Length,circumference
        ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        model1 = joblib.load("treee.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka mangga dinyatakan : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

