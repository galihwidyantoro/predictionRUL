import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve


#read data
uji = pd.read_excel('C://Users/Lenovo/Documents/1. Tugas Kuliah/SMT 8 YEAY/PROYEK AKHIR/DATA/UJI/Kejadian 11.xlsx', 
                   sheet_name=['Uji','RUL'])

df_uji = uji.get('Uji')
RUL_true = uji.get('RUL')

#load feats
with open('feats.pickle', 'rb') as a:
    feats = pickle.load(a)

#normalize
sc = MinMaxScaler(feature_range=(0,1)) 
df_uji[feats] = sc.transform(df_uji[feats])

#load feats1
with open('feats1.pickle', 'rb') as a:
    feats1 = pickle.load(a)

#load coef linear
with open('coeflinear.pickle', 'rb') as a:
    koef_linear = pickle.load(a)

#create hi linear
df_uji['HI'] = df_uji[feats1].dot(koef_linear)

#create var waktu data uji
window = 5
df_uji_HI = df_uji.groupby('Kejadian')['HI'].rolling(window = window).mean()
df_uji_HI = df_uji_HI.reset_index()
df_uji_HI.dropna(inplace = True)
df_uji_HI.drop(['level_1'], axis = 1, inplace = True)
df_uji_HI['Waktu'] = df_uji_HI.groupby('Kejadian').cumcount()+1

#load data latih HI polinomial
with open('hipoli.pickle', 'rb') as a:
    df_latih = pickle.load(a)

#load koefisien polinomial
with open('coefpoli.pickle', 'rb') as a:
    koef_poli = pickle.load(a)

#euclidean score
from math import*
 
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
        
        df_uji_fit = df_uji_fit.append({'Kejadian':i, 'Model': j, 'Nilai ED': nilai_ED, 'Panjang_RUL': total_life},
                                         ignore_index = True)

#pilih nilai euclidean terkecil
terkecil = df_uji_fit.groupby('Kejadian')['Nilai ED'].nsmallest(1).reset_index()['level_1']
df_uji_fit = df_uji_fit.iloc[terkecil]

#hitung RUL
panjang_latih = df_uji_fit.Panjang_RUL
panjang_uji = float(df_uji.Waktu.max())
hitung_RUL = RUL_true.copy()
hitung_RUL['RUL_Prediksi'] = int(panjang_latih - panjang_uji)
hitung_RUL['Eror'] = hitung_RUL.RUL_Asli - hitung_RUL.RUL_Prediksi
