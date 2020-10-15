
######### Final Analysis ###############

import numpy as np
import pandas as pd

sms = pd.read_csv('sms_clean2.csv')

sms = sms[["Type", "SMS"]]

# Change type of message (spam or ham) into Ham
sms = sms.rename(columns={"Type": "Ham"})

# convert Ham into a binary response ham =1, spam = 0
sms["Ham"] = np.where(sms["Ham"] == "ham", 1, 0)

# majority are ham messages 
sms["Ham"].mean()
# 0.8659368269921034

# remove nan values 
sms = sms.dropna() 

###################### Training the model ########################
from sklearn.model_selection import train_test_split

X = sms["SMS"] # the features we train on
ylabels = sms["Ham"] # the labels we use to test against

# test size here is only 30% of dataset
Xtrain, Xtest, ytrain, ytest = train_test_split(X, ylabels, test_size = 0.3, random_state=0)

# Use logistic regression as a classifier to build the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

################# Bag of Words Vectorization ###################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
#Fit the count vectorizer to the training data 
bvect = CountVectorizer().fit(Xtrain)

bvect.get_feature_names()[::2000]

len(bvect.get_feature_names())

# transform the documents in the training data to a document-term matrix
Xtrain_bvectorized = bvect.transform(Xtrain)
Xtrain_bvectorized

# train the vectorized data with logistic regression to make a model
model.fit(Xtrain_bvectorized, ytrain)

#predict the transformed test documents 
predictions = model.predict(bvect.transform(Xtest))

# get the feature names as numpy array
feature_names = np.array(bvect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

####################### Checking Accuracy #####################################3

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(ytest, predictions))
print("Precision:",metrics.precision_score(ytest, predictions))
print("Recall:",metrics.recall_score(ytest, predictions))

print('AUC: ', roc_auc_score(ytest, predictions))

#Check the model with a test
print(model.predict(bvect.transform(['Call now to redeem a free cash prize',
                                    'Hey, call me back I have a prize for you'])))

# For Bag of Words in Logistic

"""
Accuracy: 0.9838323353293413
Precision: 0.9835841313269493
Recall: 0.9979181124219292
AUC:  0.9465573094860738
[0 1]
"""


################## Term Frequency-Inverse Document Frequency  ##########################

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
tvect = TfidfVectorizer(min_df=8).fit(Xtrain)
len(tvect.get_feature_names())

X_train_tvectorized = tvect.transform(Xtrain)

model = LogisticRegression()
model.fit(X_train_tvectorized, ytrain)

predictions = model.predict(tvect.transform(Xtest))

## Look at the important features
feature_names = np.array(tvect.get_feature_names())

sorted_tfidf_index = X_train_tvectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


####################### Checking Accuracy #####################################3
from sklearn import metrics

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(ytest, predictions))
print("Precision:",metrics.precision_score(ytest, predictions))
print("Recall:",metrics.recall_score(ytest, predictions))

print('AUC: ', roc_auc_score(ytest, predictions))

#Check the model with a test
print(model.predict(tvect.transform(['Call now to redeem a free cash prize',
                                    'Hey, call me back I have a prize for you'])))

## For tf-idf vectorization

"""
Accuracy: 0.9706586826347305
Precision: 0.9696356275303644
Recall: 0.9972241498959056
AUC:  0.9003587998387825
[0 1]
"""


############################# N-grams ###########################################

# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
nvect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(Xtrain)

X_train_nvectorized = nvect.transform(Xtrain)

len(nvect.get_feature_names())


model = LogisticRegression()
model.fit(X_train_nvectorized, ytrain)

predictions = model.predict(nvect.transform(Xtest))

# Look at the features
feature_names = np.array(nvect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


####################### Checking Accuracy #####################################3

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(ytest, predictions))
print("Precision:",metrics.precision_score(ytest, predictions))
print("Recall:",metrics.recall_score(ytest, predictions))

print('AUC: ', roc_auc_score(ytest, predictions))

 #Check the model with a test
print(model.predict(nvect.transform(['Call now to redeem a free cash prize',
                                    'Hey, call me back I have a prize for you'])))

## For n-grams vectorization

"""
Accuracy: 0.9820359281437125
Precision: 0.9828884325804244
Recall: 0.9965301873698821
AUC:  0.9436799408465132
[0 1]
"""

############## Area under the curve receiver operating characteristics ##################

from sklearn.metrics import roc_auc_score
print('AUC: ', roc_auc_score(ytest, predictions))

###### graphing the AUROC
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(ytest, predictions)
roc_auc = metrics.auc(fpr, tpr)

# plotting
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

############################# Multinomial Naive Bayes Classification #################################

#Fit the count vectorizer to the training data 
vect = CountVectorizer().fit(Xtrain)

vect.get_feature_names()[::2000]

len(vect.get_feature_names())

# transform the documents in the training data to a document-term matrix
Xtrain_vectorized = vect.transform(Xtrain)
Xtrain_vectorized

# Post-Vectorization (transformation)

from sklearn import naive_bayes

# this is the bayes classifier
clfrNB = naive_bayes.MultinomialNB()

# train the data with Naive Bayes 
clfrNB.fit(Xtrain_vectorized, ytrain)

# after the training then we want to predict
predicted_labels = clfrNB.predict(vect.transform(Xtest))

# Once we have the prediction we can see how well we did 
# here we do the f1 score and we are micro averaging 
metrics.f1_score(ytest, predicted_labels, average='micro')
# 0.9832335329341317


####################### Checking Accuracy #####################################3

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(ytest, predicted_labels))
print("Precision:",metrics.precision_score(ytest, predicted_labels))
print("Recall:",metrics.recall_score(ytest, predicted_labels))

print('AUC: ', roc_auc_score(ytest, predicted_labels))

#Check the model with a test
print(clfrNB.predict(vect.transform(['Call now to redeem a free cash prize',
                                    'Hey, call me back I have a prize for you'])))
# Now it works
print(clfrNB.predict(vect.transform(['Call now to redeem a free cash prize',
                                    'Hey, call me back I have something for you'])))


"""
Accuracy: 0.9832335329341317
Precision: 0.9896049896049897
Recall: 0.9909784871616932
AUC:  0.9627381518777898
[0 0]
[0 1]
"""

################## Using SKlearns SVM classifier ###############################

from sklearn import svm

# Call the SVC classifier; 
# Using default radial basis function kernal
# C is the penalty parameter for soft margin, the decision boundary
# gamma is parameter for hyperplane fitting  
clfrSVM = svm.SVC(kernel='rbf', gamma=0.1, C=2) 

# then train it the same as Naive Bayes 
clfrSVM.fit(Xtrain_vectorized, ytrain)
predicted_labels = clfrSVM.predict(vect.transform(Xtest))

# Check the accuracy 
from sklearn.metrics import accuracy_score
accuracy_score(ytest, predicted_labels)
#0.9844311377245509

####################### Checking Accuracy #####################################3

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(ytest, predicted_labels))
print("Precision:",metrics.precision_score(ytest, predicted_labels))
print("Recall:",metrics.recall_score(ytest, predicted_labels))

print('AUC: ', roc_auc_score(ytest, predicted_labels))

 #Check the model with a test
print(clfrSVM.predict(vect.transform(['Call now to redeem a free cash prize',
                                    'Hey, call me back I have a prize for you'])))

"""
Accuracy: 0.9844311377245509
Precision: 0.9835953520164047
Recall: 0.9986120749479528
AUC:  0.9469042907490856
[0 1]
"""





































