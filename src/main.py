#!/usr/bin/env python3

from preprocessing import get_data
from strategies import Strategies
from sklearn import cross_validation as cv
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
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
additional_metrics = True
selected_strategy = Strategies.KNeighbors

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

# Bagging Meta-classifier
if (selected_strategy == Strategies.Bagging):
	if (adjustment):
		parameters = {'base_estimator': [None, KNeighborsClassifier],
					  'max_samples': [0.25, 0.5, 0.75, 1],
					  'max_features': [0.25, 0.5, 0.75, 1],
					  'bootstrap': [True, False], 'n_jobs': [-1], 
					  'bootstrap_features': [True, False]}
	else:
		clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=1, 
		                                             weights='uniform', n_jobs=-1))

#==============================================================================

# Trees
# Random Forest
if (selected_strategy == Strategies.RandomForest):
	if(adjustment):
		parameters = {}
	else:
		clf = RandomForestClassifier()
# Decision Tree
if (selected_strategy == Strategies.DecisionTree):
	if(adjustment):
		parameters = {}
	else:
		clf = DecisionTreeClassifier()

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

#==============================================================================

'''
if (additional_metrics):
	# Plot of a ROC curve for a specific class
	plt.figure()
	plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
'''

#==============================================================================

print('Time elapsed: {}'.format(timedelta(seconds=(time() - start_time))))
