# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 07:41:16 2019

@author: thanb
"""


#eisagwgi aparaititwn vivliothiwn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Importing dataset
beacons = pd.read_csv('beacons_dataset.csv', delimiter = ";")


X = beacons.iloc[:,:]


pd.set_option("display.max_rows",100)


#change "room" variable to lowercase
X["room"] = X["room"].str.lower()
#checking classes of "room" variable 
times = X["room"].value_counts().index

X[X["room"] == "luvingroom"]
    

#correcting variables "luvingroom" and  "dinerroom" to "livingroom"
kakes_times =  X[X["room"] == "luvingroom"]['room'].index.values 


X[X['room'] == 'luvingroom']
for i in kakes_times: 
    X.at[i,'room'] = "livingroom"
    


kakes_times =  X[X["room"] == "dinerroom"]['room'].index.values 
for i in kakes_times:
     X.at[i,'room'] = "livingroom"


#we try to correct the remain features

#we make categories of the features "room" 
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


#we change the name based on the room
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


#testing categories
X["part_id"].value_counts()

X[X["part_id"]   == 'test']
kakes_times = X[X["part_id"]   == 'test'].index.values
X = X.drop(kakes_times)

pd.set_option("display.max_rows",1000)
X["part_id"].value_counts()

#we delete values with string value as part_id
diaforetikoi_anthrwpoi = X["part_id"].value_counts().index


diaforetikoi_anthrwpoi= diaforetikoi_anthrwpoi.sort_values(ascending = False)




#we make a list with values we want to delete

times_gia_diagrafi = diaforetikoi_anthrwpoi[0:14]

times_gia_diagrafi=times_gia_diagrafi.insert(14,"12_3")
times_gia_diagrafi=times_gia_diagrafi.insert(15,"124")
times_gia_diagrafi=times_gia_diagrafi.insert(16,"123.")



#we delete values with with bad "part_id" value

for i in times_gia_diagrafi:
    kakes_times = X[X["part_id"]   == i].index.values
    X = X.drop(kakes_times)


diaforetikoi_anthrwpoi = X["part_id"].value_counts().index.sort_values()

#we make the array with every room percentages
data = np.zeros((len(diaforetikoi_anthrwpoi),5))
pososta_diaforetikwn = pd.DataFrame(data, columns = ['livingroom','bedroom', 'kitchen','outdoor','bathroom'],index = diaforetikoi_anthrwpoi)


diaforetikes_times_anthrwpou = X[X["part_id"] == "3601"]['room'].value_counts()
diaforetikes_times_anthrwpou.sum()
diaforetikes_times_anthrwpou.index[0]
diaforetikes_times_anthrwpou[0]
pososta_diaforetikwn["livingroom"][diaforetikoi_anthrwpoi[0]]

#filing the array

for j in diaforetikoi_anthrwpoi:
   
    diaforetikes_times_anthrwpou = X[X["part_id"] == j]['room'].value_counts()
    
    for i in   range(0,len(diaforetikes_times_anthrwpou)):
         
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



#importing preprocess clinical dataset from part A
clinical_for_concat = pd.read_csv('clinical_preprocessed.csv', delimiter = ",")

new_col = pososta_diaforetikwn.index.astype(int)

#constructing new id column
pososta_diaforetikwn['part_id'] = new_col

#we inner join the two datasets based on "part_id"
teliko_dataset = pd.merge(pososta_diaforetikwn,clinical_for_concat,how='inner', on='part_id' )

#we delete "part_id" column
teliko_dataset = teliko_dataset.drop(['part_id'], axis =1)

column_names = teliko_dataset.columns

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


#we see that best value is 3 from eblow method

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters  for 2 diamensions
plt.scatter(X[y_kmeans == 0 , 0], X[y_kmeans == 0 , 1], s = 100, c = 'red')  #y_means is cluster id
plt.scatter(X[y_kmeans == 1 , 0], X[y_kmeans == 1 , 1], s = 100, c = 'blue')
plt.scatter(X[y_kmeans == 2 , 0], X[y_kmeans == 2 , 1], s = 100, c = 'green')
#plt.scatter(X[y_kmeans == 3 , 0], X[y_kmeans == 3 , 1], s = 100, c = 'cyan', label = 'Careless')
#plt.scatter(X[y_kmeans == 4 , 0], X[y_kmeans == 4 , 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters ')
plt.legend()
plt.show()



































