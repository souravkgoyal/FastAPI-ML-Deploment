import pandas as pd
import numpy as np
### Train Test Split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv(r'C:\\Users\\saurabh\\Desktop\\Coursera Python\\KMV Partners\\BankNote_Authentication.csv')

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()