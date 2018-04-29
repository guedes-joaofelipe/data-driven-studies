# Credits: https://github.com/Elhamkesh/Breast-Cancer-Scikitlearn/blob/master/CancerML.ipynb

#%%
# Importing Libraries
import pandas as pd 
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier     #KNN
from sklearn.linear_model import LogisticRegression    #Logistic Regression
from sklearn.tree import DecisionTreeClassifier        #Decision Tree
from sklearn.ensemble import RandomForestClassifier    #Random Forest
from sklearn.neural_network import MLPClassifier       #Neural Network
from sklearn.svm import SVC                            #SVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve

%matplotlib inline

#%% 
# Importing Dataset
dataset = load_breast_cancer()
print (dataset.DESCR)

#%% 
# Exploring Dataset
print (dataset.data.shape)

#%%
# Initializing Metrics Vectors
algorithms_names = np.array([])
acc_train = np.array([])
acc_test = np.array([])
prec_train = np.array([])
prec_test = np.array([])
rec_train = np.array([])
rec_test = np.array([])
auc_train = np.array([])
auc_test = np.array([])

# Defining Auxiliary Functions 

def print_metrics(y_true_train, y_predicted_train, y_true_test, y_predicted_test):
    print("Accuracy of the training set: {:3f}".format(accuracy_score(y_true_train, y_predicted_train)))
    print("Accuracy of the test set: {:3f}".format(accuracy_score(y_true_test, y_predicted_test)))
    print("Precision of the training set: {:3f}".format(precision_score(y_true_train, y_predicted_train)))
    print("Precision of the test set: {:3f}".format(precision_score(y_true_test, y_predicted_test)))
    print("Recall of the training set: {:3f}".format(recall_score(y_true_train, y_predicted_train)))
    print("Recall of the test set: {:3f}".format(recall_score(y_true_test, y_predicted_test)))
    print("AUC of the training set: {:3f}".format(roc_auc_score(y_true_train, y_predicted_train)))
    print("AUC of the test set: {:3f}".format(recall_score(y_true_test, y_predicted_test)))
    

#%%
#----------KNN Classifier 
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, stratify=dataset.target, random_state=66)

training_accuracy = []
test_accuracy = []

#try KNN for different k nearest neighbor from 1 to N
max_n_neighbors = 20
neighbors_setting = range(1,max_n_neighbors)

for n_neighbors in neighbors_setting:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
 
plt.plot(neighbors_setting,training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting,test_accuracy, label='Accuracy of the test set')
plt.xticks(neighbors_setting)
plt.grid()
plt.title("Accuracy x Number of Neighbors")
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()

#%%
#by looking at plot, best result accurs when n_neighbors is 6

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
training_accuracy.append(knn.score(X_train, y_train))
test_accuracy.append(knn.score(X_test, y_test))

prediction_train = knn.predict(X_train)
prediction_test = knn.predict(X_test)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#%% Saving Results
algorithms_names = np.append(algorithms_names, "KNN")
acc_train = np.append(acc_train, accuracy_score(y_train, prediction_train))
acc_test = np.append(acc_test, accuracy_score(y_test, prediction_test))
prec_train = np.append(prec_train, precision_score(y_train, prediction_train))
prec_test = np.append(prec_test, precision_score(y_test, prediction_test))
rec_train = np.append(rec_train, recall_score(y_train, prediction_train))
rec_test = np.append(rec_test, recall_score(y_test, prediction_test))
auc_train = np.append(auc_train, roc_auc_score(y_train, prediction_train))
auc_test = np.append(auc_test, roc_auc_score(y_test, prediction_test))

#%%
#----------------Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, stratify=dataset.target, random_state=42)

log_reg_model = LogisticRegression(class_weight='balanced')
#log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

prediction_train = log_reg_model.predict(X_train)
prediction_test = log_reg_model.predict(X_test)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#It seems as it does better than KNN
#%%
algorithms_names = np.append(algorithms_names, "LogReg")
acc_train = np.append(acc_train, accuracy_score(y_train, prediction_train))
acc_test = np.append(acc_test, accuracy_score(y_test, prediction_test))
prec_train = np.append(prec_train, precision_score(y_train, prediction_train))
prec_test = np.append(prec_test, precision_score(y_test, prediction_test))
rec_train = np.append(rec_train, recall_score(y_train, prediction_train))
rec_test = np.append(rec_test, recall_score(y_test, prediction_test))
auc_train = np.append(auc_train, roc_auc_score(y_train, prediction_train))
auc_test = np.append(auc_test, roc_auc_score(y_test, prediction_test))

#%%
#----------------- Decision Tree
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42)

training_accuracy = []
test_accuracy = []

max_dep = range(1,max_n_neighbors)

for md in max_dep:
    tree = DecisionTreeClassifier(max_depth=md,random_state=0)
    tree.fit(X_train,y_train)
    training_accuracy.append(tree.score(X_train, y_train))
    test_accuracy.append(tree.score(X_test, y_test))
 
plt.plot(max_dep,training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting,test_accuracy, label='Accuracy of the test set')
plt.xticks(max_dep)
plt.grid()
plt.title("Accuracy x Max Depth")
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.legend()

# By having larger max_depth (>5), we overfit the model into training data, 
# so the accuracy for training set become but the accuracy for test set 
# decrease

# other parameters than can work with:
# - min_samples_leaf, max_sample_leaf
# - max_leaf_node

# by looking at plot, best result accurs when max_depth is 3
#%%
tree = DecisionTreeClassifier(max_depth=3,random_state=0)
tree.fit(X_train,y_train)
training_accuracy.append(tree.score(X_train, y_train))
test_accuracy.append(tree.score(X_test, y_test))

#%%
print('Feature importances: {}'.format(tree.feature_importances_))
type(tree.feature_importances_)

#Feature Importance
n_feature = dataset.data.shape[1]
plt.barh(range(n_feature), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_feature), dataset.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Decision Tress perform well and we don't need to standardize features
# But as you see, it can easilly overfit
#%%

prediction_train = tree.predict(X_train)
prediction_test = tree.predict(X_test)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#It seems as it does better than KNN
#%%
algorithms_names = np.append(algorithms_names, "Tree")
acc_train = np.append(acc_train, accuracy_score(y_train, prediction_train))
acc_test = np.append(acc_test, accuracy_score(y_test, prediction_test))
prec_train = np.append(prec_train, precision_score(y_train, prediction_train))
prec_test = np.append(prec_test, precision_score(y_test, prediction_test))
rec_train = np.append(rec_train, recall_score(y_train, prediction_train))
rec_test = np.append(rec_test, recall_score(y_test, prediction_test))
auc_train = np.append(auc_train, roc_auc_score(y_train, prediction_train))
auc_test = np.append(auc_test, roc_auc_score(y_test, prediction_test))

#%%
# ---------------- Random Forests
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train,y_train)

#you can tune parameter such as:
# - n_job (how many cores)(n_job=-1 => all cores)
# - max_depth
# - max_feature

prediction_train = forest.predict(X_train)
prediction_test = forest.predict(X_test)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#%%
#Feature Importance
n_feature = dataset.data.shape[1]
plt.barh(range(n_feature), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_feature), dataset.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Random Forest perform well and we don't need to standardize features
# Better than DT because of randomization
# It may not work well with sparse data

#%%
algorithms_names = np.append(algorithms_names, "Forest")
acc_train = np.append(acc_train, accuracy_score(y_train, prediction_train))
acc_test = np.append(acc_test, accuracy_score(y_test, prediction_test))
prec_train = np.append(prec_train, precision_score(y_train, prediction_train))
prec_test = np.append(prec_test, precision_score(y_test, prediction_test))
rec_train = np.append(rec_train, recall_score(y_train, prediction_train))
rec_test = np.append(rec_test, recall_score(y_test, prediction_test))
auc_train = np.append(auc_train, roc_auc_score(y_train, prediction_train))
auc_test = np.append(auc_test, roc_auc_score(y_test, prediction_test))

#%%
# ------------- Neural Network
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

prediction_train = mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

print_metrics(y_train, prediction_train, y_test, prediction_test)

print('The max per each feature:\n{}'.format(dataset.data.max(axis=0)))

#let's improve on NN

#%%
#1- Scaling X data
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

prediction_train = mlp.predict(X_train_scaled)
prediction_test = mlp.predict(X_test_scaled)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#%%
mlp

#%%
#2- change alpha
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(X_train_scaled,y_train)
prediction_train = mlp.predict(X_train_scaled)
prediction_test = mlp.predict(X_test_scaled)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#%%
#we can play around with other hyper parameter to improve the performance
plt.figure(figsize=(10,5))
plt.imshow(mlp.coefs_[0],interpolation='None',cmap='GnBu')
plt.yticks(range(30),dataset.feature_names)
plt.xlabel('Colums in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()

# by looking at the heatmap it seems as from "smoothness error" 
# till "fractal dimention error" does not play a huge role, 
# also "mean smoothness"

# NN can get better result in larger datasets
# we can tune a lot of parameter
# but data may need pre-processing

#other library for NN:
# theano
# keras
# tensorflow


#%%
algorithms_names = np.append(algorithms_names, "MLP")
acc_train = np.append(acc_train, accuracy_score(y_train, prediction_train))
acc_test = np.append(acc_test, accuracy_score(y_test, prediction_test))
prec_train = np.append(prec_train, precision_score(y_train, prediction_train))
prec_test = np.append(prec_test, precision_score(y_test, prediction_test))
rec_train = np.append(rec_train, recall_score(y_train, prediction_train))
rec_test = np.append(rec_test, recall_score(y_test, prediction_test))
auc_train = np.append(auc_train, roc_auc_score(y_train, prediction_train))
auc_test = np.append(auc_test, roc_auc_score(y_test, prediction_test))

#%%
# --------- SVM (Support Vector Machine)
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

svm= SVC()
svm.fit(X_train, y_train)

prediction_train = svm.predict(X_train)
prediction_test = svm.predict(X_test)

print_metrics(y_train, prediction_train, y_test, prediction_test)

#it mean we overfit into our train dataset
# we can change hyper parameter to improve the model
# one way it to apply scaling

#%%
plt.plot(X_train.min(axis=0), 'o', label='Min')
plt.plot(X_train.max(axis=0), 'v', label='Max')
plt.xlabel('Feature Index')
plt.ylabel('Feature Magnitude in Log Scale')
plt.yscale('log')
plt.legend(loc='upper right')

# We can see that there are huge diffrence between min and max 
# and between diffrent features

#%%
min_train = X_train.min(axis=0)
range_train = (X_train - min_train).max(axis=0)

X_train_scaled = (X_train - min_train)/range_train
X_test_scaled = (X_test - min_train)/range_train

print('Min per feature\n{}'.format(X_train_scaled.min(axis=0)))
print('Max per feature\n{}'.format(X_train_scaled.max(axis=0)))

#%%
svm = SVC()
svm.fit(X_train_scaled, y_train)

prediction_train = svm.predict(X_train_scaled)
prediction_test = svm.predict(X_test_scaled)

print_metrics(y_train, prediction_train, y_test, prediction_test)

# we did much better now, but now we are underfitting
# to fix it we need change hyper parameters

#%%
svm = SVC(C=1000)
svm.fit(X_train_scaled, y_train)

prediction_train = svm.predict(X_train_scaled)
prediction_test = svm.predict(X_test_scaled)

print_metrics(y_train, prediction_train, y_test, prediction_test)

# For SVM:
# can work well on high dimensional data with smaller sample size
# but don't perform well on high dim with lots of sample (>100K)
# DT or RF can be better choice, because they require less/no preprocessing of data, easier to understand and visualize


#%%
algorithms_names = np.append(algorithms_names, "SVM")
acc_train = np.append(acc_train, accuracy_score(y_train, prediction_train))
acc_test = np.append(acc_test, accuracy_score(y_test, prediction_test))
prec_train = np.append(prec_train, precision_score(y_train, prediction_train))
prec_test = np.append(prec_test, precision_score(y_test, prediction_test))
rec_train = np.append(rec_train, recall_score(y_train, prediction_train))
rec_test = np.append(rec_test, recall_score(y_test, prediction_test))
auc_train = np.append(auc_train, roc_auc_score(y_train, prediction_train))
auc_test = np.append(auc_test, roc_auc_score(y_test, prediction_test))

# --- Plotting All Metrics
#%%

width = 0.2
x_axis = np.arange(len(algorithms_names))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols = 2, figsize = (10,8))

train_color = '#008D6D'
test_color = '#DA5400'

ax1.set_ylim([0.9, 1])
ax1.set_xticklabels(np.append(np.array(["Blank"]), algorithms_names))
ax1.bar(x_axis-width/2, acc_train, width = 0.2, color = train_color)
ax1.bar(x_axis+width/2, acc_test, width = 0.2, color = test_color)
ax1.set_title("Accuracy")
ax1.legend(["Train", "Test"])

ax2.set_ylim([0.9, 1])
ax2.set_xticklabels(np.append(np.array(["Blank"]), algorithms_names))
ax2.bar(x_axis-width/2, prec_train, width = 0.2, color = train_color)
ax2.bar(x_axis+width/2, prec_test, width = 0.2, color = test_color)
ax2.set_title("Precision")
ax2.legend(["Train", "Test"])

ax3.set_ylim([0.9, 1])
ax3.set_xticklabels(np.append(np.array(["Blank"]), algorithms_names))
ax3.bar(x_axis-width/2, rec_train, width = 0.2, color = train_color)
ax3.bar(x_axis+width/2, rec_test, width = 0.2, color = test_color)
ax3.set_title("Recall")
ax3.legend(["Train", "Test"])

ax4.set_ylim([0.9, 1])
ax4.set_xticklabels(np.append(np.array(["Blank"]), algorithms_names))
ax4.bar(x_axis-width/2, auc_train, width = 0.2, color = train_color)
ax4.bar(x_axis+width/2, auc_test, width = 0.2, color = test_color)
ax4.set_title("AUC")
ax4.legend(["Train", "Test"])
