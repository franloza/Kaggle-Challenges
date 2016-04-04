#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_data
from strategies import Strategies
from sklearn import cross_validation as cv
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import RandomizedPCA
from time import time
from datetime import timedelta

start_time = time()

# Options
adjustment = False
dimension_reduc = False
additional_metrics = False
submission = False
selected_strategy = Strategies.Bagging

#==============================================================================

# Getting the data
series = get_data()

#==============================================================================

# Dimensionality Reduction
if (dimension_reduc):
	series.data = RandomizedPCA(n_components=150).fit_transform(series.data)
	print('Reduced Data size: {}'.format(series.data.shape))

#==============================================================================

# Support Vector Machines
# Linear SVC
if (selected_strategy == Strategies.LinearSVC):
	clf = LinearSVC()
	if (adjustment):
		parameters = {'C': [1, 10, 100, 1000], 'dual': [True, False],
		              'tol': [1e-4, 3e-3, 1e-3], 'max_iter': [50,100, 300, 500, 1000]}
	else:
		clf.set_params(max_iter=500, tol=0.0003, C=1, dual=True)

# SVC (Non-linear kernels)
if (selected_strategy == Strategies.SVC):
	clf = SVC()
	if (adjustment):
		parameters = [{'C': [1, 10, 100], 'kernel': ['rbf', 'sigmoid']},
					  {'C': [1, 10, 100], 'kernel': ['poly'],
					  'degree': [2, 3, 4, 5, 6]}]
	else:
		clf.set_params(kernel='rbf', C=100)

#==============================================================================

# KNeighbors / RadiusNeighbors
if (selected_strategy == Strategies.KNeighbors):
	clf = KNeighborsClassifier()
	if (adjustment):
		parameters = {'n_neighbors': [1,3,5],  'p': [1, 2],
			      	  'weights': ['uniform', 'distance'],
			      	  'algorithm': ['ball_tree', 'kd_tree', 'brute']}
	else:
		clf.set_params(n_neighbors=1, weights='uniform', algorithm='ball_tree',
		               p=2, n_jobs=-1)

if (selected_strategy == Strategies.RadiusNeighbors):
	clf = RadiusNeighborsClassifier()
	if (adjustment):
		parameters = {'radius': [1, 2, 3, 5, 10, 25],
					  'weights': ['uniform', 'distance']}
	else:
		clf.set_params(radius=1, weights='distance')

#==============================================================================

# Bagging Meta-Classifier
if (selected_strategy == Strategies.Bagging):
	clf = BaggingClassifier()
	if (adjustment):
		parameters = {'base_estimator': [None, RandomForestClassifier],
					  'n_estimators' : [3, 5, 10, 30, 50, 100]}
	else:
		clf.set_params(n_estimators=10, 
		               base_estimator=RandomForestClassifier(n_estimators=200, 
		                                        			   n_jobs=-1))

#==============================================================================

# Trees
# Random Forest
if (selected_strategy == Strategies.RandomForest):
	clf = RandomForestClassifier()
	if(adjustment):
		parameters = {}

# Decision Tree
if (selected_strategy == Strategies.DecisionTree):
	if(adjustment):
		parameters = {}
	else:
		clf = DecisionTreeClassifier(n_jobs=-1, n_estimators=200, max_features=0.9)

#==============================================================================

# Gaussian Bayes
if (selected_strategy == Strategies.GaussianBayes):
	if(adjustment):
		parameters = {}
	else:
		clf = GaussianNB()

#==============================================================================

# Adjustment
if (adjustment):
	skf = cv.StratifiedKFold(series.target, n_folds=3, shuffle=True)
	grid_search = GridSearchCV(clf, parameters, scoring='roc_auc', cv=skf, n_jobs=-1)
	grid_search.fit(series.data, series.target)
	print('Best score: {:.3f}%'.format(grid_search.best_score_ * 100))
	print('Best parameters set: {}'.format(grid_search.best_params_))

	if (submission):
		test = pd.read_csv("../data/test.csv")
		test_id = test.ID
		#test = test.drop(["ID"],axis=1)
		probs = grid_search.predict_proba(test)
		submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
		submission.to_csv("../data/submission.csv", index=False)

#==============================================================================

# 10-fold validation with given parameters & estimator
if (not adjustment):
	skf = cv.StratifiedKFold(series.target, n_folds=2, shuffle=True)
	scores = cv.cross_val_score(clf, series.data, series.target,
	                            cv=skf, scoring='roc_auc', n_jobs=-1)

	print('Scores are: ')
	print('Min: {:.3f}%  Max: {:.3f}%  Avg: {:.3f}% (+/- {:.2f}%)'
	      .format(scores.min() * 100, scores.max() * 100, scores.mean() * 100,
	              scores.std() * 200))

if (submission):
	test = pd.read_csv("../data/test.csv")
	test_id = test.ID
	#test = test.drop(["ID"],axis=1)
	prediction = clf.fit(series.data, series.target).predict(test)
	submission = pd.DataFrame({"ID":test_id, "TARGET": prediction[:,1]})
	submission.to_csv("../data/submission.csv", index=False)

#==============================================================================

if (additional_metrics):
	# ROC CURVE

	# Parameters
	test_size = .5 #Prop. of validation examples

	# Shuffle and split training and test sets
	X_train, X_test, y_train, y_test = cv.train_test_split(series.data[:-1],
	 				series.target[:-1], test_size= test_size,random_state=0)
	y_score = clf.fit(X_train, y_train).predict(X_test)

	# Compute ROC curve and ROC area
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)

	# Plot of a ROC curve for a specific class
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

#==============================================================================

print('Time elapsed: {}'.format(timedelta(seconds=(time() - start_time))))
