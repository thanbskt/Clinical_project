# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:20:19 2019

@author: thanb
"""
#eisagwgi aparaititwn vivliothiwn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 



#eisagwgi twn dataset
beacons = pd.read_csv('beacons_dataset.csv', delimiter = ";")
clinical = pd.read_csv('clinical_dataset.csv', delimiter = ";")


#reordering twn sthlwn etsi wste stin teleutaia stili na exoume tin exartimeni metavliti 'fried'
list_of_columns = clinical.columns
print(list_of_columns)
new_list_col = ['part_id', 'gender', 'age', 'hospitalization_one_year',
       'hospitalization_three_years', 'ortho_hypotension', 'vision',
       'audition', 'weight_loss', 'exhaustion_score', 'raise_chair_time',
       'balance_single', 'gait_get_up', 'gait_speed_4m',
       'gait_optional_binary', 'gait_speed_slower', 'grip_strength_abnormal',
       'low_physical_activity', 'falls_one_year', 'fractures_three_years',
       'bmi_score', 'bmi_body_fat', 'waist', 'lean_body_mass',
       'screening_score', 'cognitive_total_score', 'memory_complain', 'sleep',
       'mmse_total_score', 'depression_total_score', 'anxiety_perception',
       'living_alone', 'leisure_out', 'leisure_club', 'social_visits',
       'social_calls', 'social_phone', 'social_skype', 'social_text',
       'house_suitable_participant', 'house_suitable_professional',
       'stairs_number', 'life_quality', 'health_rate',
       'health_rate_comparison', 'pain_perception', 'activity_regular',
       'smoking', 'alcohol_units', 'katz_index', 'iadl_grade',
       'comorbidities_count', 'comorbidities_significant_count',
       'medication_count', 'fried']

clinical = clinical[new_list_col]



#i teleutaia stili tha einai kai i stili me to dianisma y me tin exartimeni metavliti
#i stili x tha exei tis anexertites metavlites
X = clinical.iloc[:, :-1]
y = clinical.iloc[:, -1]



#arxika diagrafoume tis stiles pou aforoun gia tin paragwgi tou "fried"
X = X.drop(["weight_loss","exhaustion_score","gait_speed_slower","grip_strength_abnormal","low_physical_activity"], axis =1)


#tsekaroume na doume asunithistes times se kathe mia apo tis katigories

list_col = X.columns







#kanoume nan tis megales times stin stili hospitalization three_years
X['hospitalization_three_years'].value_counts()
X[X['hospitalization_three_years'] == 999]
X["hospitalization_three_years"][24] = None
X["hospitalization_three_years"][77] = None


# kanoume nan tis megales times

#i stili "raise_chair_time"exei polles adeies kai tin diagrafooume

X = X.drop(["raise_chair_time"], axis =1)

X[X['balance_single'] =='test non realizable'  ]

#diagrafoume ti stili 'balance_single' epeidi exei polles times pou den einai swstes

X = X.drop(["balance_single"], axis =1)

#vriskoume tis times pou einai asunithista megales kai tis adeiazoume tin timi i timi NULL
kakes_times =  X[X["gait_get_up"].max() == X["gait_get_up"]]['gait_get_up'].index.values
for i in kakes_times:
     X['gait_get_up'][i] = None


kakes_times =  X[X["gait_speed_4m"].max() == X["gait_speed_4m"]]['gait_speed_4m'].index.values
for i in kakes_times:
     X['gait_speed_4m'][i] = None



kakes_times =  X[X["falls_one_year"].max() == X["falls_one_year"]]['falls_one_year'].index.values
for i in kakes_times:
     X['falls_one_year'][i] = None

kakes_times =  X[X["fractures_three_years"].max() == X["fractures_three_years"]]['fractures_three_years'].index.values
for i in kakes_times:
     X['fractures_three_years'][i] = None


#i timi auti epeidi einai megali polu ginetai NULL
X['bmi_score'][537] = None

X['bmi_body_fat'][490] = None



X["lean_body_mass"][490] = None
#diagrafoume tin stili 'memory complain' epeidi exoume polla mising values
X = X.drop(["memory_complain"], axis =1)


#diagrafoume tin stili 'cognitive_total_score' epeidi exoume polla mising values
X = X.drop(["cognitive_total_score"], axis =1)



#exoume timi me adeia timi kai epeidi einai katigoriki timi tin diagrafoume apo ta X kai Y
kakes_times = X[X["sleep"].isnull()].index.values
X = X.drop(kakes_times)
y = y.drop(kakes_times)


#omoiws gia tin stili living_alone
kakes_times = X[X["living_alone"].isnull()].index.values
X = X.drop(kakes_times)
y = y.drop(kakes_times)






kakes_times =  X[X['social_visits'] == 999]['social_visits'].index.values
for i in kakes_times:
     X['social_visits'][i] = None


kakes_times =  X[X["social_calls"].max() == X["social_calls"]]['social_visits'].index.values
for i in kakes_times:
     X['social_calls'][i] = None




kakes_times =  X[X["social_skype"].max() == X["social_skype"]]['social_skype'].index.values
for i in kakes_times:
     X['social_skype'][i] = None



kakes_times =  X[X["social_text"].max() == X["social_text"]]['social_text'].index.values
for i in kakes_times:
     X['social_text'][i] = None


X[X["house_suitable_participant"].isnull()]
#edw paratiroume pws oi stiles house_suitable_participant
#kai house_suitable_proffesional exoun polles adeis times kai einai katigorikes opote tha tis diagrapsoume
X = X.drop(["house_suitable_participant","house_suitable_professional"], axis =1)



kakes_times = X[X["health_rate"].isnull()].index.values
X = X.drop(kakes_times)
y = y.drop(kakes_times)



kakes_times = X[X["activity_regular"].isnull()].index.values
X = X.drop(kakes_times)
y = y.drop(kakes_times)


kakes_times = X[X["smoking"].isnull()].index.values
X = X.drop(kakes_times)
y = y.drop(kakes_times)


kakes_times =  X[X["alcohol_units"].max() == X["alcohol_units"]]['alcohol_units'].index.values
for i in kakes_times:
     X['alcohol_units'][i] = None



#diagrafoume tin stili "katz_index" epeidi exei polles idies times
X = X.drop(["katz_index"], axis =1)

#allazoume tis stiles etsi wste na exoume tis arxi tis katigorikes metavlites
list_col = X.columns
list_col = ['health_rate','health_rate_comparison', 'activity_regular','vision',
               'audition', 'sleep','smoking','gender', 'ortho_hypotension', 'gait_optional_binary',
               'living_alone','leisure_club',
               'age','hospitalization_one_year',
               'hospitalization_three_years', 'gait_get_up', 'gait_speed_4m',
               'falls_one_year', 'fractures_three_years', 'bmi_score', 'bmi_body_fat',
               'waist', 'lean_body_mass', 'screening_score',
               'mmse_total_score', 'depression_total_score', 'anxiety_perception',
                'leisure_out',  'social_visits',
               'social_calls', 'social_phone', 'social_skype', 'social_text',
               'stairs_number', 'life_quality',  'pain_perception', 'alcohol_units', 'iadl_grade', 'comorbidities_count',
               'comorbidities_significant_count', 'medication_count','part_id']

X = X[list_col]


#vriskoume tis stiles me missing values
stiles_me_missing_values = []
count = 0
for i in list_col :
    if len(X[X[i].isnull()]) != 0:
       stiles_me_missing_values.append(count)
    count = count +1



#kanoume ton pinaka array apo data frame
new_list_col = X.columns
new_list_col
X=X.values
#exetazoume poies stiles exoun missintg values


#gemizoume ta missing values twn stilwn pou rikame prin
for i in stiles_me_missing_values :
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
    X[:, stiles_me_missing_values] = imputer.fit_transform(X[:,stiles_me_missing_values])


#kanoume arithmous tis katigories stis prwtes times
for i in range(0,12):
    labelencoder_X = LabelEncoder()
    X [:,i] =labelencoder_X.fit_transform(X[:, i])

#arxika ftiaxnoume dummy variables gia tin kathe katigoria
#kai meta diagrafoume tin pwrti stili gia kathe metavliti/stili pou kaname labeling wste na apofugoume to dummy variabe trap
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)


X = X[:, 1:]

ct = ColumnTransformer([('encoder', OneHotEncoder(), [4])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = np.delete(X,4,1)

ct = ColumnTransformer([('encoder', OneHotEncoder(), [8])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = np.delete(X,8,1)



ct = ColumnTransformer([('encoder', OneHotEncoder(), [11])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = np.delete(X,11,1)


ct = ColumnTransformer([('encoder', OneHotEncoder(), [13])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = np.delete(X,13,1)

ct = ColumnTransformer([('encoder', OneHotEncoder(), [15])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = np.delete(X,15,1)

ct = ColumnTransformer([('encoder', OneHotEncoder(), [17])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = np.delete(X,17,1)

dataset_gia_enwsi = X
new_list_col
new_list_col = ['health_rate_dummy_1','health_rate_dummy_2','health_rate_dummy_3','health_rate_dummy_4',
                'health_rate_comparison_dummy_1','health_rate_comparison_dummy_2','health_rate_comparison_dummy_3','health_rate_comparison_dummy_4',
                'activity_regular_dummy_1','activity_regular_dummy_2','activity_regular_dummy_3',
                'vision_dummy_1','vision_dummy_2','audition_dummy_1','audition_dummy_2',
                'sleep_dummy_1','sleep_dummy_2','smoking_dummy_1','smoking_dummy_2',
               'gender', 'ortho_hypotension','gait_optional_binary', 'living_alone',
               'leisure_club', 'age','hospitalization_one_year', 'hospitalization_three_years',
               'gait_get_up', 'gait_speed_4m', 'falls_one_year',
               'fractures_three_years', 'bmi_score', 'bmi_body_fat', 'waist',
               'lean_body_mass', 'screening_score', 'mmse_total_score',
               'depression_total_score', 'anxiety_perception', 'leisure_out',
               'social_visits', 'social_calls', 'social_phone', 'social_skype',
               'social_text', 'stairs_number', 'life_quality', 'pain_perception',
               'alcohol_units', 'iadl_grade', 'comorbidities_count',
               'comorbidities_significant_count', 'medication_count', 'part_id']

new_dataset_gia_enwsi = pd.DataFrame(data = dataset_gia_enwsi , columns =new_list_col )

#new_dataset_gia_enwsi.to_csv('clinical_preprocessed.csv')

#afou exoume katisei to dataset mporoume
# na diagraposume tin stili me to part_id afou den prosferei kapoia pliroforia parapanw gia ton classifier
X = np.delete(X,53,1)



#omoiws kai gia to dianusma y
y=y.values
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)


#ftiaxnoume train kai test splits me 0.2 analogia
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#kanonikopoioume ta dedomena wste na einai stin idia vasi
from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#applying kernel PCA
from sklearn.decomposition import PCA
#we ll use 'rbf' the gausian kernel
pca = PCA(n_components =16 )
X_train = pca.fit_transform(X_train)
X_test =  pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_



plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('clinical Dataset Explained Variance')
plt.show()

#afou exoume dei to diagramma gia ta components
#mporoume na valoume san components mia timi anamesa sto 15 kai 20 wste na exasfalizoume peripou to 50%





#ulopoioume random forest algorithmo me xrisi 10 decision trees
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#upologizoume gia to X_test
y_pred = classifier.predict(X_test)

#ftiaxnoume to confusion matrix kai upologisoume tin akriberia tou montelou
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


acc = (cm[0,0] + cm[1,1] + cm[2,2]  )/107 * 100
print("i akriveia tou random forest montelou einai:"+str(acc) +"%")


#omoiws gia logistic regression
from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(random_state = 0)
classifier_logistic.fit(X_train , y_train)

y_pred = classifier_logistic.predict(X_test)

#conclusions matrix
cm_logistic = confusion_matrix(y_test, y_pred)


acc = (cm_logistic[0,0] + cm_logistic[1,1] + cm_logistic[2,2]  )/107 * 100
print("i akriveia tou logistic regression montelou einai:"+str(acc) +"%")

#naive bayes classifier
# Fitting Naives bayes  to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier_naive_bayes.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_naive_bayes = confusion_matrix(y_test, y_pred)

acc = (cm_naive_bayes[0,0] + cm_naive_bayes[1,1] + cm_naive_bayes[2,2]  )/107 * 100
print("i akriveia tou naive bayes  montelou einai:"+str(acc) +"%")



#knn classifier
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p =5)
classifier_knn.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier_knn.predict(X_test)

# Making the Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred)

acc = (cm_knn[0,0] + cm_knn[1,1] + cm_knn[2,2]  )/107 * 100
print("i akriveia tou knn montelou einai:"+str(acc) +"%")

#svm classifier
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_SVM.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier_SVM.predict(X_test)

# Making the Confusion Matrix
cm_svc = confusion_matrix(y_test, y_pred)

acc = (cm_svc[0,0] + cm_svc[1,1] + cm_svc[2,2]  )/107 * 100
print("i akriveia tou svm montelou einai:"+str(acc) +"%")





















