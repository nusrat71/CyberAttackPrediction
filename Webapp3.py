import pandas as pd

import streamlit as st
from sklearn import *
from PIL import Image
st.header('Cyber Attack: Industry Prediction')
image=Image.open('C:/Users/nsrtk/PycharmProjects/pythonProject/Cyber.png')
st.image(image,caption='',use_column_width=True)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\\nsrtk\\file1(1).csv")
df.isnull().sum()
df=df.dropna()
####
st.subheader('Data Information')
st.dataframe(df)
st.write(df.describe())


import math
max_request=df['Number of requests'].max()
min_request=df['Number of requests'].min()
difference=max_request-min_request
mid_value=math.floor(difference/2)
import math
mid_mid_value=math.floor(difference/3)
first_margin_diff=mid_value-mid_mid_value
first_margin=mid_value-first_margin_diff
second_margin=mid_value+first_margin_diff
# Allocate happiness to bins
binned = pd.cut(df["Number of requests"], bins = [min_request,first_margin,second_margin,max_request], labels = [1,2,3])


# Add the binned values as a new categorical feature
df["severity"] = binned






df=df.drop(["Date/Time"],axis=1)
df.drop_duplicates()



from sklearn.preprocessing import MinMaxScaler

# Creating an instance of the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()

# Scaling the Price column of the created dataFrame and storing
# the result in ScaledPrice Column
df[["Number of requests"]] = scaler.fit_transform(df[["Number of requests"]])

#########


#most_freq=df['Industry'].value_counts().idxmax()
most_freq=df['Industry'].value_counts().idxmax()
df['Industry'].values[df['Industry'].value_counts()[df['Industry']] < 100] =most_freq
df['Industry'].values[df['Industry']=='Sports'] ='Sports & Gaming'
df['Industry'].values[df['Industry']=='Gaming'] ='Sports & Gaming'

df["Industry_Attack_Severity"]=df["Industry"]

df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Travel'] =1
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Financial Services'] =2
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Business'] =3
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Computing & IT'] =4
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Retail'] =5
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Education'] =6
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Telecom and ISPs'] =7
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Law & Government'] =8
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Healthcare'] =9
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Lifestyle'] =10
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Automotive'] =11
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Sports & Gaming'] =12
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Entertainment & the Arts'] =13
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Society'] =14
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='News'] =15
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Gambling'] =16
df['Industry_Attack_Severity'].values[df['Industry_Attack_Severity']=='Food & Beverages'] =17

##########
df_original=df

#######

#########

Client_ui=st.selectbox('Select Attack Tool:',
                                df['Client'].unique())
Type_of_Attack_ui= st.selectbox('Select Type of Attack:',
                                df['Type_of_Attack'].unique())
Source_Country_ui = st.selectbox('Select Source Country:',
                                df['Source_Country'].unique())
Destination_Country_ui = st.selectbox('Select Destination Country:',
                                df['Destination_Country'].unique())
 # group=df.groupby(['Destination_Country','Industry','Client']).get_group(("United States", "Sports","Hacking Tool")).Type_of_Attack.value_counts().unstack(fill_value=0)


user_data={'Attack Tool':Client_ui,
           'Type_of_attack':Type_of_Attack_ui,
            'Source Country':Source_Country_ui,
             'Destination_Country':Destination_Country_ui

   }


user_input=pd.DataFrame(user_data,index=[0])
 ##################
#user_input=get_user_input()
st.subheader('User Input:')
st.write(user_input)
#########
st.header("Prediction of Industry: ")
######


# Import the libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import pandas as pd

df = df_original
df=df_original
df=df.drop(['Industry'], axis=1)
df = df.drop(['severity'], axis=1)
df = df.drop(['Number of requests'], axis=1)

print(df)

Destination_Country_v = Destination_Country_ui
Source_Country_v= Source_Country_ui
Client_v = Client_ui
Type_of_Attack_v = Type_of_Attack_ui

Attack_Severity = 0

df.loc[0] = [Source_Country_v,Destination_Country_v,Type_of_Attack_v, Client_v, Attack_Severity]  # adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index()  # sorting by index

# most_freq=df['Type_of_Attack'].value_counts().idxmax()
# df['Type_of_Attack'].values[df['Type_of_Attack']=="null"] =most_freq


# Create an object for Base N Encoding
encoder = ce.BaseNEncoder(cols=['Client', 'Source_Country', 'Destination_Country', 'Type_of_Attack'], return_df=True, base=5)

# Fit and Transform Data
df = encoder.fit_transform(df)

X = df.drop("Industry_Attack_Severity", axis=1).values
y = df.Industry_Attack_Severity.values
y = pd.to_numeric(y, errors="coerce")
y = y.astype('int')


# Split our training and testing sets
Xtest, Xtrain, ytest, ytrain = train_test_split(X, y, random_state=None, test_size=0.7, shuffle=False)
# Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=None,test_size=0.3,shuffle=False)


# Create a Pipeline, use StandarScaler and Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

M1 = Pipeline([
    ('scl', StandardScaler()),
    ('lr',
     RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=16, min_samples_split=2, min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                            oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,
                            class_weight=None))
])
M1.fit(Xtrain, ytrain)
ypred = M1.predict(Xtest)
print(ypred[0])
user_val = ypred[0]

df=df_original
Destination_Country_v=Destination_Country_ui
Source_Country_v=Source_Country_ui
Type_of_Attack_v=Type_of_Attack_ui
Client_v=Client_ui
Attack_Severity=user_val
Industry='Null'

df=df.drop(['severity'], axis=1)
df=df.drop(['Number of requests'], axis=1)


df.loc[0] = [Source_Country_v,Destination_Country_v,Industry,Type_of_Attack_v,Client_v,Attack_Severity]  # adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index()  # sorting by index

#most_freq=df['Type_of_Attack'].value_counts().idxmax()
#df['Type_of_Attack'].values[df['Type_of_Attack']=="null"] =most_freq

print(df)

#########

#Import the libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import pandas as pd

#Create an object for Base N Encoding
#Create an object for Base N Encoding

encoder= ce.BaseNEncoder(cols=['Client','Type_of_Attack','Destination_Country','Source_Country'],return_df=True,base=5)


#Fit and Transform Data
df=encoder.fit_transform(df)
X=df.drop("Industry",axis=1).values
y=df.Industry.values



# Split our training and testing sets

Xtest,Xtrain,ytest,ytrain= train_test_split(X,y,random_state=None,test_size=0.7,shuffle=False)



# Create a Pipeline, use StandarScaler and Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

M1=Pipeline([
    ('scl',StandardScaler()),
    ('lr',RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=16,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None))
])
M1.fit(Xtrain,ytrain)
ypred=M1.predict(Xtest)
#print(ypred)
score = M1.score(Xtest, ytest)
print(score)
###Prediction

print(ypred[0])

st.write(ypred[0])

