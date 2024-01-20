#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd


# In[3]:


idx=(2015,2016,2017,2018,2019,2020,2021)
dt=(150,160,170,180,200,130,90)


# In[7]:


pd.Series( data=dt,index=idx)


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.plot(pd.Series( data=dt,index=idx))


# In[10]:


sözlük={2015:150,2016:160,2017:170,2018:180,2019:200,2020:130,2021:90}
pd.Series(sözlük)


# In[15]:


plt.plot(pd.Series(sözlük), "green")
plt.show()


# In[28]:


tablo=pd.Series( data=dt,index=idx)
pd.DataFrame(tablo,columns= ["ton"])


# In[29]:


tablo=tablo.reset_index()


# In[30]:


tablo


# In[33]:


tablo=tablo.rename(columns={"index":"Yıl"})


# In[34]:


tablo


# In[139]:


import numpy as np


# In[43]:


a=np.array(1)       #nokta
b=np.array([1])     #vektör
c=np.array([[1]])   #matris
d=np.array([[[1]]]) #tensor


# In[38]:


a.ndim #ndim kac boyutlu olduğunu gösterir.


# In[40]:


b.ndim


# In[42]:


c.ndim


# In[44]:


d.ndim


# In[52]:


veri=np.array([[1],[2],[3]])


# In[54]:


pd.DataFrame(veri)


# In[55]:


veri=np.array([[1,"ayşe",8000],[2,"kerim",12000],[3,"buse",15000]])


# In[65]:


tablo=pd.DataFrame(data=veri)


# In[66]:


tablo


# In[62]:


col=["sıra","isim","maas"]


# In[73]:


tablo.columns=col


# In[74]:


idx=["a","b","c"]


# In[75]:


tablo.index=idx


# In[76]:


tablo


# In[33]:


matris=np.random.randn(10,2) #rand yazınca sadece pozitif sayılar randn yazınca hem pozitif hem negatif sayılar
matris=matris*100 #0 ile 100 arası
matris=matris.round() #yuvarlıyor tam sayıya


# In[34]:


tablo=pd.DataFrame(data=matris)


# In[35]:


tablo


# In[36]:


idx=["A","B","C","D","E","F","G","H","I","J"]
col=["kolon1","kolon2"]


# In[37]:


tablo.columns=col
tablo.index=idx


# In[38]:


tablo


# In[40]:


matris=(np.random.rand(4,4)*60).round()


# In[41]:


tablo=pd.DataFrame(data=matris ,index=["A","B","C","D"],columns=["kolon 1","kolon 2","kolon 3","kolon 4"])


# In[42]:


tablo


# In[102]:


import matplotlib.pyplot as plt


# In[46]:


plt.plot(tablo)
plt.legend(tablo) #hangi vekötörün hangi değere ait oldugunu sol üstte gösterir


# In[56]:


matris=(np.random.rand(3,3)*100).round()


# In[57]:


tablo=pd.DataFrame(data=matris, index=["a","b","c"],columns=["x","y","z"])


# In[65]:


tablo


# In[58]:


tablo.loc["c"]


# In[59]:


tablo.loc[["c"]]


# In[63]:


tablo.iloc[0:1,0:2]


# In[64]:


tablo.iloc[2,1]


# In[66]:


tablo


# In[67]:


tablo.iloc[1:2,1:2]=80


# In[68]:


tablo


# In[69]:


idx1=["Ocak","Ocak","Ocak","Ocak","Şubat","Şubat","Şubat","Şubat"]
idx2=["Hafta1","Hafta2","Hafta3","Hafta4","Hafta1","Hafta2","Hafta3","Hafta4"]


# In[70]:


idx_m=list(zip(idx1,idx2))


# In[71]:


idx_m


# In[72]:


idx_m=pd.MultiIndex.from_tuples(idx_m)


# In[75]:


type(idx_m)


# In[76]:


idx_m


# In[77]:


liste=[200,350,500,650,1500,815,150,400]


# In[87]:


tablo=pd.DataFrame(data=liste,index=idx_m,columns=["Satış"])


# In[79]:


tablo


# In[82]:


tablo.loc["Ocak"]


# In[84]:


tablo.loc[["Şubat"]]


# In[85]:


tablo.loc["Şubat"].loc["Hafta2"]


# In[88]:


sözlük={"X":[5,45,np.nan,3],"Y":[np.nan,14,3,9],"Z":[4,6,12,7]}


# In[91]:


sözlük


# In[95]:


idx=["A","B","C","D"]


# In[106]:


tablo=pd.DataFrame(data=sözlük,index=idx)


# In[97]:


tablo


# In[98]:


tablo.dropna() #eksik değer olmayan satırlar


# In[99]:


tablo.dropna(axis=1) #eksik değer olmayan sütunlar axis=0 satır, axis=1 sütun


# In[108]:


tablo.fillna(10) #nan olan değer yerine eksikveri yazar


# In[109]:


plt.plot(tablo)


# In[ ]:


concat #iki tabloyu birlestirme


# In[118]:


matris1=(np.random.rand(3,3)*100).round()


# tablo1=pd.DataFrame(data=matris1, index=["A","B","C"],columns=["A","B","C"])

# In[120]:


tablo1=pd.DataFrame(data=matris1, index=["A","B","C"],columns=["A","B","C"])
tablo1


# In[121]:


matris2=(np.random.rand(3,3)*100).round()
tablo2=pd.DataFrame(data=matris2, index=["A","B","C"],columns=["A","B","C"])
tablo2


# In[124]:


tablo3=pd.concat([tablo1,tablo2])
tablo3


# In[125]:


tablo3=pd.concat([tablo1,tablo2],axis=1) #yatay
tablo3


# In[ ]:


#merge tabloları birleştiriken sadece farklı olanları alır aynıları tekrarlamaz.


# In[134]:


sözlük={"bir":["1","11","111"],"iki":["2","22","222"]}
tablo1=pd.DataFrame(data=sözlük)
tablo1


# In[135]:


sözlük2={"bir":["1","11","111"],"üç":["3","33","333"]}
tablo2=pd.DataFrame(data=sözlük2)
tablo2


# In[136]:


tablo3=pd.merge(tablo1,tablo2,on="bir")
tablo3


# In[137]:


import pandas as pd

sözlük = {"Ay":["Ocak","Ocak","Şubat",
                "Mart", "Nisan","Nisan","Nisan"], 
          "Miktar": [37543, 75530, 53208, 25476, 28578, 95001, 35746]}

tablo = pd.DataFrame(sözlük)
tablo


# In[140]:


ay=tablo.groupby("Ay")


# In[141]:


ay


# In[143]:


ay.count()


# In[144]:


ay.describe()


# In[161]:




# In[159]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Veri setini oluşturun veya yükleyin
data = {
    'Yıl': [2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Mal ve Hizmet Vergileri': [84.4, 87.9, 93.2, 99.5, 108.0, 145.5, 46.44],
    'Harçlar': [36.6, 34.1, 29.8, 35.5, 44.0, 62.5, 63.56]
}

df = pd.DataFrame(data)

# Bağımsız değişkenleri ve bağımlı değişkeni seçin
X = df[['Mal ve Hizmet Vergileri']]
y = df['Harçlar']

# Veriyi eğitim ve test setlerine bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doğrusal regresyon modelini oluşturun
model = LinearRegression()
model.fit(X_train, y_train)

# Modeli test seti üzerinde değerlendirin
y_pred = model.predict(X_test)

# Hata metriklerini hesaplayın
print('Ortalama Kare Hata:', metrics.mean_squared_error(y_test, y_pred))
print('R-kare:', metrics.r2_score(y_test, y_pred))

# Modelin katsayılarını ve intercept'ini yazdırın
print('Katsayılar:', model.coef_)
print('Intercept:', model.intercept_)

# Modelin görselleştirilmesi
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Mal ve Hizmet Vergileri')
plt.ylabel('Harçlar')
plt.title('Doğrusal Regresyon Modeli')
plt.show()


# In[162]:


ßß