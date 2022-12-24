import pandas as pd

import streamlit as st

from PIL import Image
st.header('Types of Cyber Attack Prediction')
#image=Image.open('https://github.com/nusrat71/CyberAttackPrediction/blob/main/Cyber.PNG')
#st.image(image,caption='',use_column_width=True)


import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
df=pd.read_csv("https://github.com/nusrat71/CyberAttackPrediction/blob/master/file1(1).csv")
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


most_freq=df['Source_Country'].value_counts().idxmax()
df['Source_Country'].values[df['Source_Country'].value_counts()[df['Source_Country']] < 9200] =most_freq
most_freq=df['Destination_Country'].value_counts().idxmax()
df['Destination_Country'].values[df['Destination_Country'].value_counts()[df['Destination_Country']] < 9600] =most_freq
df['Source_Country'] = df['Source_Country'] .str.strip()
from sklearn.preprocessing import MinMaxScaler

# Creating an instance of the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()

# Scaling the Price column of the created dataFrame and storing
# the result in ScaledPrice Column
df[["Number of requests"]] = scaler.fit_transform(df[["Number of requests"]])

#########
df["Attack_Severity"]=df["Type_of_Attack"]

df['Attack_Severity'].values[df['Attack_Severity']=='Automated Threat - Business Logic'] =1
df['Attack_Severity'].values[df['Attack_Severity']=='Automated Threat'] =2
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - API Violation'] =3
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - RCE/RFI'] =4
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - Path Traversal/LFI'] =5
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - XSS'] =6
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - SQLi'] =7
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - Data Leakage'] =8
df['Attack_Severity'].values[df['Attack_Severity']=='DDoS'] =9
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - Protocol Manipulation'] =10
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - Backdoor/Trojan'] =11
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - File Upload'] =12
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - SSRF'] =13
df['Attack_Severity'].values[df['Attack_Severity']=='Automated Threat - Spam'] =14
df['Attack_Severity'].values[df['Attack_Severity']=='Automated Threat - Account Takeover'] =15
df['Attack_Severity'].values[df['Attack_Severity']=='OWASP - Authentication Bypass'] =16
##########
df_original=df

#######

#########
Source_Country_ui=st.selectbox('Select Source Country:',
                                df['Source_Country'].unique())
Client_ui=st.selectbox('Select Attack Tool:',
                                df['Client'].unique())
Industry_ui = st.selectbox('Select Industry:',
                                df['Industry'].unique())
Destination_Country_ui = st.selectbox('Select Destination Country:',
                                df['Destination_Country'].unique())
 # group=df.groupby(['Destination_Country','Industry','Client']).get_group(("United States", "Sports","Hacking Tool")).Type_of_Attack.value_counts().unstack(fill_value=0)


user_data={'Source Country':Source_Country_ui,
            'Attack Tool':Client_ui,
            'Industry':Industry_ui,
             'Destination_Country':Destination_Country_ui

   }


user_input=pd.DataFrame(user_data,index=[0])
 ##################
#user_input=get_user_input()
st.subheader('User Input:')
st.write(user_input)
#########
st.header("Prediction of Types of Attack: ")
######


# Import the libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import pandas as pd

df = df_original
df = df.drop(['severity'], axis=1)
df = df.drop(['Number of requests'], axis=1)
df = df.drop(['Type_of_Attack'], axis=1)

Source_Country_v=Source_Country_ui
Destination_Country_v = Destination_Country_ui
Industry_v = Industry_ui
Client_v = Client_ui
Attack_Severity = 0

df.loc[0] = [Source_Country_v,Destination_Country_v, Industry_v, Client_v, Attack_Severity]  # adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index()  # sorting by index

# most_freq=df['Type_of_Attack'].value_counts().idxmax()
# df['Type_of_Attack'].values[df['Type_of_Attack']=="null"] =most_freq
print(df)

# Create an object for Base N Encoding
encoder = ce.BaseNEncoder(cols=['Client', 'Industry', 'Destination_Country','Source_Country'], return_df=True, base=5)

# Fit and Transform Data
df = encoder.fit_transform(df)

X = df.drop("Attack_Severity", axis=1).values
y = df.Attack_Severity.values
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
user_val = ypred[0]

df=df_original
Source_Country_v=Source_Country_ui
Destination_Country_v=Destination_Country_ui
Industry_v=Industry_ui
Type_of_Attack_v='Null'
Client_v=Client_ui
Attack_Severity=user_val

df=df.drop(['severity'], axis=1)
df=df.drop(['Number of requests'], axis=1)


df.loc[0] = [Source_Country_v,Destination_Country_v, Industry_v,Type_of_Attack_v,Client_v,Attack_Severity]  # adding a row
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
encoder= ce.BaseNEncoder(cols=['Client','Industry','Destination_Country','Source_Country'],return_df=True,base=5)

#Fit and Transform Data
df=encoder.fit_transform(df)

X=df.drop("Type_of_Attack",axis=1).values
y=df.Type_of_Attack.values



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

