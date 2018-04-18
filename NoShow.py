#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:21:58 2018

@author: Khushboo Baheti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold 
#load data
raw_data = pd.read_csv('/Users/khushboo/Desktop/pyproj/KaggleV2-May-2016.csv')

raw_data.head()

#rename columns
raw_data = raw_data.rename(columns={'Gender': 'sex', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 
                                    'Age': 'age', 'Neighbourhood': 'neighbourhood', 'Scholarship': 'scholarship', 
                                    'Hipertension': 'hypertension', 'Diabetes': 'diabetic', 'Alcoholism': 'alcoholic', 
                                    'Handcap': 'handicap', 'No-show': 'no_show'})
raw_data.head()
raw_data.describe()
#binarize columns and plotting each field with no_show

#0-Show  & 1-Noshow
print("0-Show  1-Noshow")
raw_data['no_show'] = raw_data['no_show'].map({'No': 1, 'Yes': 0})

#0-Female & 1-Male
print("0-Female  1-Male")
raw_data['sex'] = raw_data['sex'].map({'F': 0, 'M': 1})
#sex - no-show
sns.countplot(x='sex', hue='no_show', data=raw_data, palette='RdBu')
#plt.show();

#check handicap class count
sns.countplot(x='handicap', hue='no_show', data=raw_data, palette='RdBu')
#since after two in handicapped field there is no change in result so we can put 3 and 4 into 2 
raw_data['handicap'] = raw_data['handicap'].apply(lambda x: 2 if x > 2 else x)
#handicap - no-show
sns.countplot(x='handicap', hue='no_show', data=raw_data, palette='RdBu')

#get data and time
raw_data['scheduled_day'] = pd.to_datetime(raw_data['scheduled_day'], infer_datetime_format=True)
raw_data['appointment_day'] = pd.to_datetime(raw_data['appointment_day'], infer_datetime_format=True)

#age-no-show
sns.countplot(x='age',hue='no_show', data=raw_data)
#we can see in the age-noshow graph, it's not making any insight
#age is -1,0 or greater than 100.Remove outliers
raw_data.drop(raw_data[raw_data['age'] <= 0].index, inplace=True)
raw_data.drop(raw_data[raw_data['age'] >100].index, inplace=True)
raw_data.describe()


#scholarship - no-show
sns.countplot(x='scholarship',hue='no_show', data=raw_data)

#hypertension - no-show
sns.countplot(x='hypertension',hue='no_show', data=raw_data)

#diabetic - no-show
sns.countplot(x='diabetic',hue='no_show', data=raw_data)

#alcoholic - no-show
sns.countplot(x='alcoholic',hue='no_show', data=raw_data)

#neighbourhood - no-show
#http://pbpython.com/categorical-encoding.html
#https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931
encoder_neighbourhood = LabelEncoder()
raw_data['neighbourhood_enc'] = encoder_neighbourhood.fit_transform(raw_data['neighbourhood'])


#new fields we have added
#added appiontment_dayofweek and all records are having only weekday
#0-Monday & 6-Sunday
raw_data['appointment_dayofWeek'] = raw_data['appointment_day'].map(lambda x: x.dayofweek)
raw_data.head()
print("The day of the week with Monday=0, Sunday=6")
#appointment_dayofweek - no-show
sns.countplot(x='appointment_dayofWeek',hue='no_show', data=raw_data)

#Now we are adding one more field after seeing the normal distribution of Age
#1-18 = 1, 19-37 = 2, 38-55 = 3, 56-75 = 4, 75-100 = 5
raw_data['age_group'] = raw_data['age'].apply(lambda x: 1 if x>0 and x<19 else 
                                                            2 if x>18 and x<38 else 
                                                            3 if x>37 and x<56 else 
                                                            4 if x>55 and x<76 else 5)
#age-group - no-show
sns.countplot(x='age_group',hue='no_show', data=raw_data)
#group-5 is percentage of no-show is higher than others 

#we are adding one more filed Insurance_age according to the medicare and medicaid
#1-medicare & 0-medicaid
raw_data['insurance_age'] = raw_data['age'].apply(lambda x: 1 if x >= 65 else 0)
#insurance_age - no-show
sns.countplot(x='insurance_age',hue='no_show', data=raw_data)

"""when we converted the waiting time to days, we saw there were negative days.That is appointmentday< scheduled day
We dropped them and also added +1 to no.of days"""

raw_data['waiting_time'] = list(map(lambda x: x.days+1 , raw_data['appointment_day'] - raw_data['scheduled_day']))
raw_data.drop(raw_data[raw_data['waiting_time'] <= -1].index, inplace=True)
raw_data.describe()

#checking frequency of each waiting time
data = Counter(raw_data['waiting_time']) 
data.most_common() # Returns all unique items and their counts 

raw_data['waiting_time_range'] = raw_data['waiting_time'].apply(lambda x: 1 if x>=0 and x<=30 else 
                                                          2 if x>30 and x<=60 else 
                                                          3 if x>60 and x<=90 else 
                                                          4 if x>90 and x<=120 else 
                                                          5 if x>120 and x<=150 else
                                                          6 if x>150 and x<=180 else
                                                          7)
#waiting_time - no-show
sns.countplot(x='waiting_time',hue='no_show', data=raw_data)
#waiting_time_range - no-show
sns.countplot(x='waiting_time_range',hue='no_show', data=raw_data)
#4-6 waiting_time_range can be removed since it's not making any insight

raw_data['no_of_noshows'] = raw_data.groupby('PatientId')[['no_show']].transform('sum')
raw_data['total_appointment'] = raw_data.groupby('PatientId')[['no_show']].transform('count')
raw_data['risk_score'] = (raw_data.no_of_noshows / raw_data.total_appointment)

sns.countplot(x='risk_score',hue='no_show', data=raw_data)
#4-6 waiting_time_range can be removed since it's not making any insight

#remove id columns which are not needed for predictions 
raw_data.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
raw_data.head()

#corrleration matrix
sns.heatmap(raw_data.corr()) 

raw_data.drop(['scheduled_day', 'appointment_day', 'neighbourhood'], axis=1, inplace=True)

X = raw_data.drop(['no_show'], axis=1)
y = raw_data['no_show']
X.head()
y.head()
Counter(y)
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)
Counter(y_res)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)

#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
clf.feature_importances_
#score method that can judge the quality of the fit (or the prediction) on new data.
clf.score(X_test, y_test)

print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))

feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')


X = raw_data[['SMS_received','waiting_time','age','sex']]
y = raw_data['no_show']

X.head()
y.head()
Counter(y)
sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_sample(X, y)
Counter(y_res)

kf = KFold(n_splits=3,shuffle=True)


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=101)

clf = RandomForestClassifier(n_estimators=200)

for k, (train, test) in enumerate(kf.split(X, y)):
   clf.fit(X_train,y_train)
   print(clf.score(X_test, y_test))

