
try:
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
    import plost
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import copy
    from numpy import sqrt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import minmax_scale
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve
except Exception as e:
    print(e)
 
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
 
 
class FileUpload(object):
 
    def __init__(self):
        self.fileTypes = ["csv"]
 
    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        
        #st.info(__doc__)
        st.title('Prediksi Remaining Useful Life')
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
            return
        content = file.getvalue()
        df_uji = pd.read_csv(file)
        #st.dataframe(df_uji.head(10))
        
        #RUL_true = 5342

        #load feats
        #with open('feats.pickle', 'rb') as a:
        #    feats = pickle.load(a, encoding='latin1')
        feats = pd.read_pickle("feats.pickle")

        #load scaler
        #with open('scaler2.pickle', 'rb') as a:
        #    sc = pickle.load(a, encoding='latin1')
        sc = pd.read_pickle("scaler2.pickle")

        #normalize
        df_uji[feats] = sc.transform(df_uji[feats])

        #load feats1
        #with open('feats1.pickle', 'rb') as a:
        #    feats1 = pickle.load(a, encoding='latin1')
        feats1 = pd.read_pickle("feats1.pickle")

        #load coef linear
        #with open('coeflinear.pickle', 'rb') as a:
        #    koef_linear = pickle.load(a, encoding='latin1')
        koef_linear = pd.read_pickle("coeflinear.pickle")

        #create hi linear
        df_uji['HI'] = df_uji[feats1].dot(koef_linear)

        #create var waktu data uji
        window = 5
        df_uji_HI = df_uji.groupby('Kejadian')['HI'].rolling(window = window).mean()
        df_uji_HI = df_uji_HI.reset_index()
        df_uji_HI.dropna(inplace = True)
        df_uji_HI.drop(['level_1'], axis = 1, inplace = True)
        df_uji_HI['Waktu'] = df_uji_HI.groupby('Kejadian').cumcount()+1

        #load data latih
        df_latih = pd.read_pickle("data_latih.pickle")
        
        #load data latih HI polinomial
        #with open('hipoli.pickle', 'rb') as a:
        #    df_latih = pickle.load(a, encoding='latin1')
        df_latih_HI = pd.read_pickle("hipoli.pickle")

        #load koefisien polinomial
        #with open('coefpoli.pickle', 'rb') as a:
        #    koef_poli = pickle.load(a, encoding='latin1')
        koef_poli = pd.read_pickle("coefpoli.pickle")

        #euclidean score
        def euclidean_distance(x,y):
            return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

        df_uji_fit = pd.DataFrame(columns = ['Kejadian', 'Model', 'Nilai ED', 'Panjang_RUL'])

        for i in df_uji.Kejadian.unique():
            
            HI = df_uji_HI.HI[df_uji_HI.Kejadian == i]
            Waktu = df_uji_HI.Waktu[df_uji_HI.Kejadian == i]
                
            for j in koef_poli.Kejadian.unique():
                
                Theta_0 = koef_poli.Theta_0[koef_poli.Kejadian == j].values
                Theta_1 = koef_poli.Theta_1[koef_poli.Kejadian == j].values
                Theta_2 = koef_poli.Theta_2[koef_poli.Kejadian == j].values
                
                pred_HI = Theta_0 + Theta_1*Waktu + Theta_2*Waktu*Waktu
                
                nilai_ED = euclidean_distance(pred_HI,HI)
                
                total_life = df_latih.Waktu[df_latih.Kejadian == j].max()
                
                df_uji_fit = df_uji_fit.append({'Kejadian':i, 'Model': j, 'Nilai ED': nilai_ED, 'Panjang_RUL': total_life}, ignore_index = True)

        
        #pilih nilai euclidean terkecil
        terkecil = df_uji_fit.groupby('Kejadian')['Nilai ED'].nsmallest(1).reset_index()['level_1']
        df_uji_fit = df_uji_fit.iloc[terkecil]
        use = (df_uji_fit.Model).max()
        case = (df_uji_fit.Kejadian).max()

        RUL_set = pd.read_pickle("RUL.pickle")
        part_im = pd.read_pickle("datapart.pickle")
        for i in range(1,30):
            if i == case:
                RUL_asli=RUL_set.loc[i-1,'RUL_Asli']
                part_pilih=part_im.loc[i-1,'part']
        
        #hitung RUL
        panjang_latih = df_uji_fit.Panjang_RUL
        panjang_uji = float(df_uji.Waktu.max())
        RUL_Prediksi = int(panjang_latih - panjang_uji)
        RUL_jam = RUL_Prediksi/60
        RUL_jam = round(RUL_jam,1)
        RUL_hari = RUL_jam/24
        RUL_hari = round(RUL_hari,1)
        eror = RUL_asli - RUL_Prediksi

        #gabungkan data
        HIuji = df_uji_HI[["Kejadian","HI"]].reset_index()
        HIuji = HIuji.drop(['index'], axis=1)
        HIlatih = df_latih_HI[["Waktu","Kejadian","HI_poli"]].reset_index()
        HIlatih = HIlatih.drop(['index'], axis=1)
        HIlatih = HIlatih[HIlatih.Kejadian==use].reset_index()
        HIlatih = HIlatih.drop(['index'], axis=1)
        HIlatih['HI']=HIuji['HI']
        HIlatih['Prediksi']=HIlatih['HI_poli']

        st.header("Hasil Prediksi RUL")
        col1,col2,col3=st.columns(3)
        col1.metric(label="Dalam menit", value=-RUL_Prediksi, delta="Menit")
        col2.metric(label="Dalam jam",value=-RUL_jam,delta="Jam")
        col3.metric(label="Dalam hari",value=-RUL_hari,delta="Hari")
        plost.line_chart(
            data= HIlatih,
            x='Waktu',
            y=('Prediksi','HI')
        )
        st.subheader("Rekomendasi Pergantian Parts")
        st.text(part_pilih)

        file.close()
 
 
if __name__ ==  "__main__":
    helper = FileUpload()
    helper.run()