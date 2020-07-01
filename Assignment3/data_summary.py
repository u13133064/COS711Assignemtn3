import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

np.seterr(all='raise')
def label_row (row):
   if row['target'] >= 0 and row['target'] <=12 :
      return 'green'
   elif row['target'] > 12 and row['target'] <=35 :
      return 'yellow'
   elif row['target'] > 35 and row['target'] <=55 :
      return 'orange'
   elif row['target'] > 55 and row['target'] <=150 :
      return 'red'
   elif row['target'] > 150 and row['target'] <=250 :
      return 'purple'
   else:
       return 'brown'
def get_data():
    return pd.read_csv('Train.csv')  
def get(df, location):
    np.set_printoptions(precision=3, suppress=True)
    frame = df.loc[df['location'] == location].copy()
    frame['class'] = df.apply(lambda row:  label_row(row), axis=1)
    return frame

def normalize(maxv,minv,x):
    if(x< minv or x > maxv ):
        print(x)
        print(minv)
        print(maxv)

        raise NameError('HiThere')
    if ((maxv-minv) == 0):
        return x
    try:
         a = (x-minv)/(maxv-minv)
    except:
        
        print(x,minv,maxv)
    
    return (x-minv)/(maxv-minv)
def fix(temp,minv,maxv):
    i = 0
    while i != 121:
        if(np.isnan(temp[i])):
            if(i-1>=0):
                temp[i] = temp[i-1]
            else:
                save = i+1
                while(np.isnan(temp[save])):
                    save+=1
                if(np.isnan(temp[save])):
                    temp[i] = (minv+maxv)/2
                else: 
                    temp[i] =temp[save]


                
        i+=1

    i = 0
    while i != 121:
       temp[i] = normalize(minv,maxv,temp[i])
       i+=1
    return temp
def encode(location):
    ### Universal list of colors
    total = ["A", "B", "C", "D", "E"]
    
    if  location == "A":
        return normalize(4,0,0)
    if  location == "B":
        return normalize(4,0,1)
    if  location == "C":
        return normalize(4,0,2)
    if  location == "D":
        return normalize(4,0,3)
    if  location == "E":
        return normalize(4,0,4)
def createSeries(location, temps,precips,rel,dirs,spds,atmos):
    l = encode(location)
    t = fix(temps,36.53333333,13.78333333)
    #37 13
    rel = fix(rel,1.0, 0.172416667)
    p = fix(precips,45.833, 0.0)
    d = fix(dirs,359.9973829, 0.012509863)
    s = fix(spds,24.86666667, 0.123333333)
    a = fix(atmos,91.13083333, 87.25)
    series = []
    for i in range(0,121):
       series.append([l,t[i],p[i],rel[i],d[i],s[i],a[i]])
    return series
def getSequences(df):
    targets =[]
    x =[]
    for index, row in df.iterrows():
        temps = np.genfromtxt(np.array(row['temp'].split(",")))
        precips = np.genfromtxt(np.array(row['precip'].split(",")))
        rel_hums = np.genfromtxt(np.array(row['rel_humidity'].split(",")))
        wind_dirs = np.genfromtxt(np.array(row['wind_dir'].split(",")))
        wind_spds = np.genfromtxt(np.array(row['wind_spd'].split(",")))
        atmos_press = np.genfromtxt(np.array(row['atmos_press'].split(",")))
        location = row['location']
        targets.append(row['target'])
        
   
        x.append(createSeries(location, temps,precips,rel_hums,wind_dirs,wind_spds,atmos_press))
    return np.array(x), np.array(targets)

        
def getMinMax(df,col):
    maxv = 0
    minv = 1000
    for index, row in df.iterrows():
        vals = np.genfromtxt(np.array(row[col].split(",")))
        x= vals.max()
        m= vals.min()

        if (x > maxv):
            maxv =    x= vals.max()
        if (m < minv):
            minv = m
        

    return maxv, minv
df = get_data()
# print(getMinMax(df,'temp'))
# print(getMinMax(df,'precip'))
# print(getMinMax(df,'rel_humidity'))
# print(getMinMax(df,'wind_dir'))
# print(getMinMax(df,'wind_spd'))

# print(getMinMax(df,'atmos_press'))
# (36.53333333, 14.65)
# (45.833, 0.0)
# (1.0, 0.172416667)
# (359.9973829, 0.012509863)
# (24.86666667, 0.123333333)
# (91.13083333, 87.25)
#A= get(df,'A')
#df['location'].value_counts().plot(kind='pie',autopct='%1.1f%%')


#A= A[A['class'] != 'brown']
#A= A[A['class'] != 'green']
#A= A[A['class'] != 'purples']


x,y = getSequences(df)
print(y)
pickle.dump( x, open( "ProcessedX.pkl", "wb" ) )
pickle.dump( y, open( "Processedy.pkl", "wb" ) )
print("done")
