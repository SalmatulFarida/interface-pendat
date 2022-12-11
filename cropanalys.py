import streamlit as st
import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle

st.title("Applikasi Web Datamining")
st.write("""
\n Copyright : SALMATUL FARIDA 
# Crop analysis and prediction Dataset
Web ini akan menggunakan dataset Crop analysis and prediction,
web ini juga dapat menginputkan suatu data dari setiap fitur yang ada dalam dataset,
juga anda dapat melihat kumpulan data yang memungkinkan pengguna membangun model prediktif untuk merekomendasikan tanaman yang paling cocok untuk ditanam di pertanian tertentu berdasarkan berbagai parameter yang nanti akan menghasilkan label tanaman.
dari beberapa algoritma yang di sediakan dalam website ini dapat melihat akurasi yang paling terbaik
dari model algoritma tersebut.
""")

# inisialisasi data 
data = pd.read_csv("Crop_recommendation.csv")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description Data", "Preprocessing Data", "Modeling", "Implementation","Profil"])

with tab1:

    st.subheader("Deskripsi Dataset")
    st.write("""Dataset di bawah ini menjelaskan prekursor untuk kumpulan data dan pengembangannya memberikan 
    hasil (tidak divalidasi silang) untuk klasifikasi oleh sistem pakar berbasis aturan dengan versi kumpulan 
    data tersebut. Dataset ini dibuat dengan menambah kumpulan data curah hujan, iklim, dan data pupuk yang tersedia untuk India. """)

    st.write("""
    ### Want to learn more?
    - Dataset [kaggel.com](https://www.kaggle.com/code/theeyeschico/crop-analysis-and-prediction/data)
    - Github Account [github.com](https://github.com/SalmatulFarida/datamining)
    """)

    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
    st.write("""
    ### Data Understanding
    Penjelasan dari setiap fitur yang
    ada dalam dataset Crop analysis and prediction  :
    1. N - rasio kandungan Nitrogen dalam tanah
    2. P - rasio kandungan Fosfor dalam tanah
    3. K - rasio kandungan Kalium dalam tanah
    4. temperature - suhu dalam derajat Celcius
    5. humidity - kelembaban relatif dalam %
    6. ph - nilai ph tanah
    7. rainfall - curah hujan dalam mm
    """)

with tab2:
    st.subheader("Data Preprocessing")
    st.subheader("Data Asli")
    data = pd.read_csv("Crop_recommendation.csv")
    st.write(data)

    proc = st.checkbox("Normalisasi")
    if proc:

        # Min_Max Normalisasi
        from sklearn.preprocessing import MinMaxScaler
        df_for_minmax_scaler=pd.DataFrame(data, columns = ['N',	'P',	'K',	'temperature',	'humidity',	'ph','rainfall'])
        df_for_minmax_scaler.to_numpy()
        scaler = MinMaxScaler()
        df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

        st.subheader("Hasil Normalisasi Min_Max")
        df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns =['N',	'P',	'K',	'temperature',	'humidity',	'ph','rainfall'])
        st.write(df_hasil_minmax_scaler)

        st.subheader("tampil data label")
        df_label = pd.DataFrame(data, columns = ['label'])
        st.write(df_label.head())

        st.subheader("Gabung Data")
        df_new = pd.concat([df_hasil_minmax_scaler,df_label], axis=1)
        st.write(df_new)

        st.subheader("Drop fitur label")
        df_drop_site = df_new.drop(['label'], axis=1)
        st.write(df_drop_site)

        st.subheader("Hasil Preprocessing")
        df_new = pd.concat([df_hasil_minmax_scaler,df_label], axis=1)
        st.write(df_new)

with tab3:

    X=data.iloc[:,0:7].values
    y=data.iloc[:,7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("KNN")
    model2 = st.checkbox("Naive Bayes")
    model3 = st.checkbox("Random Forest")
    # model4 = st.checkbox("Ensamble Stacking")

    if model1:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNN.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)
    if model2:
        model = GaussianNB()
        filename = "gausianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    if model3:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Random Forest : ",score)
   
with tab4:
    # Min_Max Normalisasi
    from sklearn.preprocessing import MinMaxScaler
    df_for_minmax_scaler=pd.DataFrame(data, columns = ['N',	'P',	'K',	'temperature',	'humidity',	'ph','rainfall'])
    df_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

    df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ['N',	'P',	'K',	'temperature',	'humidity',	'ph','rainfall'])

    df_label = pd.DataFrame(data, columns = ['label'])

    df_new = pd.concat([df_hasil_minmax_scaler,df_label], axis=1)

    df_drop_site = df_new.drop(['label'], axis=1)

    df_new = pd.concat([df_hasil_minmax_scaler,df_label], axis=1)

    st.subheader("Parameter Inputan")
    N = st.number_input("Masukkan N :")
    P = st.number_input("Masukkan P :")
    K = st.number_input("Masukkan K :")
    temperature = st.number_input("Masukkan temperature :")
    humidity = st.number_input("Masukkan humidity:")
    ph = st.number_input("Masukkan Ph :")
    rainfall = st.number_input("Masukkan rainfall :")
    hasil = st.button("cek klasifikasi")

    # Memakai yang sudah di preprocessing
    X=df_new.iloc[:,0:7].values
    y=df_new.iloc[:,7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    if hasil:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [N, P, K, temperature, humidity, ph, rainfall]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        st.write(f"Algoritma yang digunakan adalah = Random Forest Algorithm")
        st.success(f"Hasil Akurasi : {score}")
with tab5:
    st.write("""
    \n NAMA  : SALMATUL FARIDA 
    \n NIM   : 200411100016 
    \n KELAS : PENAMBANGAN DATA A
    \n EMAIL : 200411100016@STUDUDENT.TRUNOJOYO.AC.ID
    """)
