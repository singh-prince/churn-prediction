#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import pandas as pd 
import numpy as np 

import json
import os
from numpy import array
import string
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import model_from_json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
#import matplotlib.pyplot as plt
#import seaborn as sns


# In[2]:


from flask_bootstrap import Bootstrap 
from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print('Model loaded. Check http://127.0.0.1:5000/')

# In[3]:


app = Flask(__name__)
Bootstrap(app)


# In[4]:


@app.route('/')
def index():
    return render_template('index.html')

df = pd.read_csv('churn.all')

# Define a function to visulize the features with missing values, and % of total values, & datatype
def missing_values_table(df):
     # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_type = df.dtypes
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_type], axis=1)
        
     # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'type'})
        
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n" "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

missing_values_table(df)

df['voice_mail_plan'] = df['voice_mail_plan'].map(lambda x: x.strip())
df['intl_plan'] = df['intl_plan'].map(lambda x: x.strip())
df['churned'] = df['churned'].map(lambda x: x.strip())

# yes/no -> 0/1
df['voice_mail_plan'] = df['voice_mail_plan'].map({'no':0, 'yes':1})
df.intl_plan = df.intl_plan.map({'no':0, 'yes':1})

df.churned.value_counts(normalize=True)

df.groupby('churned').mean()

from scipy.stats import ks_2samp
def run_KS_test(feature):
    dist1 = df.loc[df.churned == 'False.',feature]
    dist2 = df.loc[df.churned == 'True.',feature]
    #print(feature+':')
    #print(ks_2samp(dist1,dist2),'\n')
    
from statsmodels.stats.proportion import proportions_ztest
def run_proportion_Z_test(feature):
    dist1 = df.loc[df.churned == 'False.', feature]
    dist2 = df.loc[df.churned == 'True.', feature]
    n1 = len(dist1)
    p1 = dist1.sum()
    n2 = len(dist2)
    p2 = dist2.sum()
    z_score, p_value = proportions_ztest([p1, p2], [n1, n2])
    #print(feature+':')
    #print('z-score = {}; p-value = {}'.format(z_score, p_value),'\n')
    
from scipy.stats import chi2_contingency
def run_chi2_test(df, feature):

    dist1 = df.loc[df.churned == 'False.',feature].value_counts().sort_index().tolist()
    dist2 = df.loc[df.churned == 'True.',feature].value_counts().sort_index().tolist()
    chi2, p, dof, expctd = chi2_contingency([dist1,dist2])
    #print(feature+':')
    #print("chi-square test statistic:", chi2)
    #print("p-value", p, '\n')
    
ks_list = ['account_length','number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']

z_list = ['voice_mail_plan','intl_plan' ]

for ks_element in ks_list:
    run_KS_test(ks_element)
for z_element in z_list:
    run_proportion_Z_test(z_element)
    
ks_list = ['account_length','number_vmail_messages','total_day_minutes','total_day_calls','total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']

df.churned = df.churned.map({'False.':0, 'True.':1})

corr = df.drop(['area_code'], axis=1).corr()

# Drop some useless columns
drop_list = ['state', 'account_length', 'total_day_calls', 'total_eve_calls', 'total_night_calls','total_day_minutes', 'total_eve_minutes','total_night_minutes','voice_mail_plan', 'area_code','phone_number','total_intl_minutes', 'total_intl_calls','churned']

X = df.drop(drop_list, axis=1)
y = df.churned.values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

a1_train=X_train.values
a1_test=X_test.values

list_min_max=[]
for i in range(7):
    list_min_max.append([min(a1_train[:,i]),max(a1_train[:,i])])
    print(i,min(a1_train[:,i]),max(a1_train[:,i]))
    
for i in range(7):
    a1_train[:,i]=(a1_train[:,i]-list_min_max[i][0])/(list_min_max[i][1]-list_min_max[i][0])
    
for i in range(7):
    a1_test[:,i]=(a1_test[:,i]-list_min_max[i][0])/(list_min_max[i][1]-list_min_max[i][0])
    
# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X.columns)

#print("Feature space holds %d observations and %d features" % X_train.shape)
#print("Unique target labels:", np.unique(y_train))

def check(a,ynew):
    
    if (a==1):
            message=100+100*ynew
            call=80+80*ynew
            discount=200-200*ynew
    elif(a==2):
            message=200+200*ynew
            call=160+100*ynew
            discount=400-400*ynew
    elif(a==3):
            message=300+300*ynew
            call=240+100*ynew
            discount=600-600*ynew
    elif(a==4):
            message=400+400*ynew
            call=350+350*ynew
            discount=800-800*ynew
             
    return message,call,discount

print('Model loaded. Check http://127.0.0.1:5000/')

# In[5]:
@app.route('/predict', methods=['POST'])
def predict():
        
        # load json and create model
        #modelfile = 'models/final_model.pickle'   
        
        # Receives the input query from form
        if request.method == 'POST':
            #data = [namequery]
            a= request.form['plan']
            p= request.form['p1']
            q= request.form['q1']
            r= request.form['r1']
            s= request.form['s1']
            t= request.form['t1']
            u= request.form['u1']
            v= request.form['v1']
            
            Xnew = np.array([[p,q,r,s,t,u,v]])
            print("hi")
            
            #print(int(Xnew[0][0])-int(list_min_max[0][0]))
            #print(int(list_min_max[0][1])-int(list_min_max[0][0]))
            
            
            for i in range(0,7):
                a1=float(Xnew[:,i])-float(list_min_max[i][0])
                b1=float(list_min_max[i][1])-float(list_min_max[i][0])
                #print(a)
                #print(b)
                Xnew[:,i]=float(float(a1)/float(b1))
                
            
            print(Xnew)
            #Make a Prediction
            
         
            ynew =  model.predict(Xnew)
           
            print("predicted probability churn")
            print(ynew)
            
            #print(a)
            #message,call,discount=check(a,ynew)
            
            #result=[message,call,discount]
            message=0
            call=0
            discount=0
            
            
            if (a=="1"):
                        message=100+100*ynew
                        call=80+80*ynew
                        discount=200-200*ynew
                        #print("hi",message,call,discount)
            elif(a=="2"):
                        message=200+200*ynew
                        call=160+100*ynew
                        discount=400-400*ynew
                        #print("hi",message,call,discount)
            elif(a=="3"):
                        message=300+300*ynew
                        call=240+100*ynew
                        discount=600-600*ynew
                        #print("hi",message,call,discount)
            elif(a=="4"):
                        message=400+400*ynew
                        call=350+350*ynew
                        discount=800-800*ynew
                        #print("hi",message,call,discount)            
            
            """result=["predicted probability churn="+str(ynew),"Number_of_Messages="+str(int(message)),"Number_of_calls="+str(int(call)),"Total_discount="+str(int(discount))]"""
            
            result={'Predicted probability Not churn=':str(ynew[0][0]) ,'Number_of_Messages=':str(int(message)) ,'Number_of_calls=':str(int(call)),
                   'Total_discount=':str(int(discount))}
            
            
            
            
            return render_template('output.html', res=result)
        return None
            

# In[6]:

if __name__ == '__main__':
     # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

