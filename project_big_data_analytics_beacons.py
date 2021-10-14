# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 07:41:16 2019

@author: thanb
"""


#eisagwgi aparaititwn vivliothiwn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#kanoume import to dataset
beacons = pd.read_csv('beacons_dataset.csv', delimiter = ";")


X = beacons.iloc[:,:]


pd.set_option("display.max_rows",100)


#kanoume ola ta onomata twn room se mikra
X["room"] = X["room"].str.lower()
#afou vroume tis diaforetikes times kai to poses fores uparxoun kai to apothikeuoume stin emtavliti times
times = X["room"].value_counts().index

X[X["room"] == "luvingroom"]
    

#diorthwnoume tis times me onoma "luvingroom" kai "dinerroom" se "livingroom"
kakes_times =  X[X["room"] == "luvingroom"]['room'].index.values 


X[X['room'] == 'luvingroom']
for i in kakes_times: 
    X.at[i,'room'] = "livingroom"
    


kakes_times =  X[X["room"] == "dinerroom"]['room'].index.values 
for i in kakes_times:
     X.at[i,'room'] = "livingroom"


#prospathoume na kanoume to idio gia tis upoloipes times

#vriskoume to index stin lista times pou dimiourgisame prin kai vazoume
#ta index se 5 diaforetikes  katigories analoga me to pou ta katatasoume
bedroom = [11,18,23,29,30,44,45,46,48,49,53,74]
kitchen = [12,20,22,24,27,36,38,51,54,56,62,66,69]
livingroom = [5,6,7,8,9,10,15,16,19,21,25,26,28,31,32,34,37,39,40,42,50,57,59,64,65,70,71,72,75,76,78]
outdoor =[13,17,35,41,43,55,61,75,79]
bathroom = [14,33,47,52,58,60,63,67,68,73,77,80]

bedroom_names = []
kitchen_names = []
livingroom_names = []
outdoor_names = []
bathroom_names = []

#ftiaxnoume listes me ta diaforetika onomata twn kathe katigoriwn
for i in bedroom: 
    bedroom_names.append(times[i])

for i in kitchen: 
    kitchen_names.append(times[i])
    
for i in livingroom: 
    livingroom_names.append(times[i])   

for i in outdoor: 
    outdoor_names.append(times[i])

for i in bathroom: 
    bathroom_names.append(times[i])

len(X["room"].value_counts())


#gia oles tis diaforetikes onomasies allazoume to onoma tou "room" analoga se poia katigoria anhkoun
for s in range(0,len(bedroom_names)) :
    kakes_times =  X[X["room"] == bedroom_names[s]]['room'].index.values 
    for i in kakes_times:
        X.at[i,'room'] = "bedroom"
    

for s in range(0,len(kitchen_names)) :
    kakes_times =  X[X["room"] == kitchen_names[s]]['room'].index.values 
    for i in kakes_times:
        X.at[i,'room'] = "kitchen"
        
for s in range(0,len(livingroom_names)) :
    kakes_times =  X[X["room"] == livingroom_names[s]]['room'].index.values 
    for i in kakes_times:
        X['room'][i] = "livingroom"
        
for s in range(0,len(outdoor_names)) :
    kakes_times =  X[X["room"] == outdoor_names[s]]['room'].index.values 
    for i in kakes_times:
        X['room'][i] = "outdoor"


for s in range(0,len(bathroom_names)) :
    kakes_times =  X[X["room"] == bathroom_names[s]]['room'].index.values 
    for i in kakes_times:
        X['room'][i] = "bathroom"


#tsekaroume pali kai vlepoume an exoume katalixei stis 5 katigories
X["part_id"].value_counts()

X[X["part_id"]   == 'test']
kakes_times = X[X["part_id"]   == 'test'].index.values
X = X.drop(kakes_times)

pd.set_option("display.max_rows",1000)
X["part_id"].value_counts()

#diagrafoume kai tis upoloipes grammes pou exoun part_id kapoio string kai den tairiazei
diaforetikoi_anthrwpoi = X["part_id"].value_counts().index


diaforetikoi_anthrwpoi= diaforetikoi_anthrwpoi.sort_values(ascending = False)




#gemizoume tin lista times_gia_diagrafi me tis asxetes times twn part_id pou tha diagrapsoume
times_gia_diagrafi = diaforetikoi_anthrwpoi[0:14]

times_gia_diagrafi=times_gia_diagrafi.insert(14,"12_3")
times_gia_diagrafi=times_gia_diagrafi.insert(15,"124")
times_gia_diagrafi=times_gia_diagrafi.insert(16,"123.")



#diagrafoume tis grammes pou exoun san part_id asxeti timi

for i in times_gia_diagrafi:
    kakes_times = X[X["part_id"]   == i].index.values
    X = X.drop(kakes_times)

#apothikeuoume tous diaforetikous anthrwpous
diaforetikoi_anthrwpoi = X["part_id"].value_counts().index.sort_values()

#kataskeuazoume to dataframe me ta pososta twn anthrwpwn
data = np.zeros((len(diaforetikoi_anthrwpoi),5))
pososta_diaforetikwn = pd.DataFrame(data, columns = ['livingroom','bedroom', 'kitchen','outdoor','bathroom'],index = diaforetikoi_anthrwpoi)


diaforetikes_times_anthrwpou = X[X["part_id"] == "3601"]['room'].value_counts()
diaforetikes_times_anthrwpou.sum()
diaforetikes_times_anthrwpou.index[0]
diaforetikes_times_anthrwpou[0]
pososta_diaforetikwn["livingroom"][diaforetikoi_anthrwpoi[0]]

#gemizoun ton pinaka pososta_diaforetikwn me ta pososta tou kathena se kathe apo ta dwmatia

for j in diaforetikoi_anthrwpoi:
   # print("upologizoume ta pososta gia ton anthrwpo me part_id:" + j)
    diaforetikes_times_anthrwpou = X[X["part_id"] == j]['room'].value_counts()
    #print(diaforetikes_times_anthrwpou)
    for i in   range(0,len(diaforetikes_times_anthrwpou)):
         #print("to i einai:" + str(i))
         if diaforetikes_times_anthrwpou.index[i] == "livingroom":
             pososta_diaforetikwn["livingroom"][j]  = diaforetikes_times_anthrwpou[i]/diaforetikes_times_anthrwpou.sum()
    
         elif diaforetikes_times_anthrwpou.index[i] == "bedroom":
             pososta_diaforetikwn["bedroom"][j]  = diaforetikes_times_anthrwpou[i]/diaforetikes_times_anthrwpou.sum()
                 
         
         elif diaforetikes_times_anthrwpou.index[i] == "kitchen":
             pososta_diaforetikwn["kitchen"][j]  = diaforetikes_times_anthrwpou[i]/diaforetikes_times_anthrwpou.sum()
         
         
         elif diaforetikes_times_anthrwpou.index[i] == "outdoor":
             pososta_diaforetikwn["outdoor"][j]  = diaforetikes_times_anthrwpou[i]/diaforetikes_times_anthrwpou.sum()
         
         
         elif diaforetikes_times_anthrwpou.index[i] == "bathroom":
             pososta_diaforetikwn["bathroom"][j]  = diaforetikes_times_anthrwpou[i]/diaforetikes_times_anthrwpou.sum()



#kanoume import to dataset pou proekupse apo to part A
clinical_for_concat = pd.read_csv('clinical_preprocessed.csv', delimiter = ",")

# pairnoume tis times tou index kai epeidi einai se morfi string tis kanoume int
new_col = pososta_diaforetikwn.index.astype(int)

#topothetoume sto dataframe mia stili me to part_id pou ousiastika eixame sto index
pososta_diaforetikwn['part_id'] = new_col

#gia na vgaloume to enwmeno data set kanoume inner join twn duo data frame xrisimopoiwntas tin stili part_id ws koino stoixeio
teliko_dataset = pd.merge(pososta_diaforetikwn,clinical_for_concat,how='inner', on='part_id' )

#diagrafoume tin stili "part_id" pou pleon den mas xreiazetai allo
teliko_dataset = teliko_dataset.drop(['part_id'], axis =1)

column_names = teliko_dataset.columns
#teliko_dataset = teliko_dataset.values
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
teliko_dataset = sc.fit_transform(teliko_dataset)




#applying kernel PCA
from sklearn.decomposition import KernelPCA
#we ll use 'rbf' the gausian kernel
kpca = KernelPCA(n_components = 2, kernel = 'rbf' )
X = kpca.fit_transform(teliko_dataset)






'''
#Fitting the PCA algorithm with our Data
from sklearn.decomposition import KernelPCA
pca = KernelPCA().fit(teliko_dataset)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()
'''


#using the elbow method to find the optimal numer of clusters
from sklearn.cluster import KMeans 
wcss = []
for i in range (1,11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0 )
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)#i methodos inertia mas dinei to wcss sto kontinotero kentro 
plt.plot(range(1, 11), wcss) 
plt.title('The nubmer Elbow method')    
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#vlepoume oti gia ta sugkekrimena dedomena exoume 3 clusters

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)





#visualizing the clusters  for 2 diamensions
plt.scatter(X[y_kmeans == 0 , 0], X[y_kmeans == 0 , 1], s = 100, c = 'red')  #y_means einaio arithmnos tou cluster pou anikei
plt.scatter(X[y_kmeans == 1 , 0], X[y_kmeans == 1 , 1], s = 100, c = 'blue')
plt.scatter(X[y_kmeans == 2 , 0], X[y_kmeans == 2 , 1], s = 100, c = 'green')
#plt.scatter(X[y_kmeans == 3 , 0], X[y_kmeans == 3 , 1], s = 100, c = 'cyan', label = 'Careless')
#plt.scatter(X[y_kmeans == 4 , 0], X[y_kmeans == 4 , 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters ')
plt.legend()
plt.show()



































