"""Uji ablasi fitur premium, terpisah dari notebook UTS Kelompok 6.

Penggunaan (dari root repository, dengan CSV yang sudah disertakan):
    python scripts/Analisis_Ablasi_Data_Leakage_Kelompok_6.py

Memerlukan: numpy, pandas, scikit-learn.
Hanya melatih ulang dua model baseline dalam eksperimen TERPISAH;
notebook asli, file .keras, dan output UTS tidak diubah.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(description="Ablasi dampak kebocoran fitur premium (dua baseline)")
root = Path(__file__).resolve().parents[1]
parser.add_argument('--data', type=Path, default=root / 'data/medical_insurance.csv', help='CSV proyek lokal')
parser.add_argument('--out', type=Path, default=root / 'results', help='Folder CSV hasil')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(args.data)
base=pd.get_dummies(raw.drop_duplicates().reset_index(drop=True),columns=['gender','discount_eligibility','region'],drop_first=True,dtype=int)
y=base['expenses'];safe=base.drop(columns=['expenses','premium']);leak=base.drop(columns=['expenses'])
itr,itmp=train_test_split(np.arange(len(base)),test_size=.30,random_state=42)
ival,ite=train_test_split(itmp,test_size=.50,random_state=42)
rows=[]
for label,X,numeric in [('Tanpa premium (aman)',safe,['age','bmi','children']),('Dengan premium (bocor)',leak,['age','bmi','children','premium'])]:
  Xtr,Xte=X.iloc[itr].copy(),X.iloc[ite].copy()
  scaler=StandardScaler().fit(Xtr[numeric]);Xtr[numeric]=scaler.transform(Xtr[numeric]);Xte[numeric]=scaler.transform(Xte[numeric])
  ytr=np.log1p(y.iloc[itr]);yte=y.iloc[ite]
  methods={'Linear Regression':LinearRegression(), 'Random Forest':RandomForestRegressor(n_estimators=300,min_samples_leaf=3,random_state=42,n_jobs=-1)}
  for name,model in methods.items():
    model.fit(Xtr,ytr);pr=np.expm1(model.predict(Xte))
    rows.append({'Kondisi':label,'Model':name,'MAE':mean_absolute_error(yte,pr),'RMSE':np.sqrt(mean_squared_error(yte,pr)),'R2':r2_score(yte,pr)})
res=pd.DataFrame(rows)
res.to_csv(args.out/'Hasil_Ablasi_Kelompok_6.csv',index=False)
print('Data awal:',len(raw),'Data bersih:',len(base),'Train:',len(itr),'Validation:',len(ival),'Test:',len(ite))
print(res.to_string(index=False,float_format=lambda a:f'{a:.6f}'))
print('CSV disimpan:', args.out/'Hasil_Ablasi_Kelompok_6.csv')
