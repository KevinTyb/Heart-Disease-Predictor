# Imports

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[79]:


pd.options.mode.chained_assignment = None  # default='warn'
data = pd.read_csv('C:/Users/kevin/Documents/heart.csv')
data.head()


# In[80]:


#Check for missing data
data.isnull().sum()


# In[81]:


# check all variables are integers
data.info()


# In[82]:


# compare correlation of each variable to output
data.corr()['output'].sort_values(ascending = False)


# In[83]:


pd.crosstab(data.age,data.output).plot(kind="bar",figsize=(20,6))
plt.title('Heart Attack Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[399]:


# Set x and y variables
# drop data we are training on to prevent data leakage
X = data.drop('output', axis = 1) 
y = data ['output']


# In[400]:


# Create a Train test split model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Logistic Regression Classifier

# In[401]:


from sklearn.linear_model import LogisticRegression

# instantiate the model 
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=log.predict(X_test)


# In[402]:


from sklearn.model_selection import cross_val_score
cmLog = confusion_matrix(y_test, logreg.predict(X_test))

TN = cmLog[0][0]
TP = cmLog[1][1]
FN = cmLog[1][0]
FP = cmLog[0][1]

print(cmLog)

print('Model Test Accuracy = {}' .format( (TP+TN)/ (TP+TN+FN+FP ) ) )


# In[403]:


# scale data for increased performance

sc = StandardScaler()
mm = MinMaxScaler()

sc_X_train, sc_X_test = sc.fit_transform(X_train), sc.transform(X_test)
mm_X_train, mm_X_test = mm.fit_transform(X_train), sc.transform(X_test)


# In[404]:


modelDev(LogisticRegression, sc_X_train, y_train, sc_X_test, y_test)


# In[405]:


modelDev(LogisticRegression, mm_X_train, y_train, mm_X_test, y_test)


# In[406]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, X, y, cv=5)
scores


# In[407]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# # Random Forest Classifier Algorithm

# In[475]:


#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
forest.fit(X_train, y_train)


# In[476]:


# test accuracy of RF

forestModel = forest
forestModel.score(X_train, y_train)


# In[455]:


cmForest = confusion_matrix(y_test, forestModel.predict(X_test))

TN2 = cmForest[0][0]
TP2 = cmForest[1][1]
FN2 = cmForest[1][0]
FP2 = cmForest[0][1]

print(cmForest)

print('Model Test Accuracy = {}' .format( (TP2+TN2)/ (TP2+TN2+FN2+FP2 ) ) )


# In[457]:


# Random Forest Cross Validation
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(forestModel, X, y, cv=5)
scores2


# In[458]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))


# In[464]:


params = {'max_depth': [2,3,5,10,20], 'min_samples_leaf': [5,10,20,50,100,200], 'n_estimators': [10,25,30,50,100,200]
}


# In[465]:


from sklearn.model_selection import GridSearchCV


# In[466]:


grid_search = GridSearchCV(estimator=forest,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[467]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[468]:


grid_search.best_score_


# # K-Nearest Neighbour Classifier Algorithm

# In[490]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print("{} NN Score: {:.2f}%".format(2, clf.score(X_test, y_test)))


# In[496]:


k_value = []
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn.fit(X_train, y_train)
    k_value.append(knn.score(X_test, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

max_acc = max(scoreList)
print("Maximum KNN Score is {:.2f}".format(max_acc))


# In[479]:


#Confusion Matrix
cmNeighbor = confusion_matrix(y_test, clf.predict(X_test))

TN3 = cmNeighbor[0][0]
TP3 = cmNeighbor[1][1]
FN3 = cmNeighbor[1][0]
FP3 = cmNeighbor[0][1]

print(cmNeighbor)

print('Model Test Accuracy = {}' .format( (TP3+TN3)/ (TP3+TN3+FN3+FP3 ) ) )


# In[480]:


# Cross Validation
scores3 = cross_val_score(clf, X, y, cv=5)
scores3


# In[481]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))


# # Decision Tree Classification Algorithm

# In[418]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[419]:


dtc.score(X_test, y_test)


# In[424]:


cmTree = confusion_matrix(y_test, dtc.predict(X_test))

TN4 = cmTree[0][0]
TP4 = cmTree[1][1]
FN4 = cmTree[1][0]
FP4 = cmTree[0][1]

print(cmTree)

print('Model Test Accuracy = {}' .format( (TP4+TN4)/ (TP4+TN4+FN4+FP4 ) ) )


# In[425]:


scores4 = cross_val_score(dtc, X, y, cv=5)
scores4


# In[422]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores4.mean(), scores4.std()))

