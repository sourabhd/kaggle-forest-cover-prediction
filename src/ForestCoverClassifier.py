from __future__ import print_function
import ConfigParser
import numpy as np
import os
import utils
import tempfile
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
import pdb
import gc
import time
import cPickle as pickle
from copy import deepcopy
import array
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
import sys
import pandas as pd
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
#from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import LeaveOneOut
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier

class ForestCoverClassifier:
    """Forest Cover Classifer"""

    def __init__(self, runinfo, settings='../settings.ini'):
        """Initialize object"""

        self.runinfo = runinfo
        print('Numpy:', np.__version__)
        print('Pandas:', pd.__version__)
        print('Scikit-Learn:', sklearn.__version__)
        self.config = ConfigParser.SafeConfigParser()
        self.config.read(settings)
        self.randomSeed = int(self.config.get('system', 'random_seed'))
        self.dsetName = self.config.get('dataset', 'dset_name')
        self.trainFile = self.config.get('dataset', 'train_file')
        self.testFile = self.config.get('dataset', 'test_file')
        self.scratchLoc = self.config.get('system', 'scratch_loc')
        self.outDir = self.config.get('system', 'out_dir')
        print(self.dsetName)
        np.set_printoptions(threshold=np.nan)   # Print complete array


    def readDataset(self):
	
	train_df = pd.read_csv(self.trainFile)
	test_df = pd.read_csv(self.testFile)

	#print(train_df.columns)
	#print(train_df.head())	
	#print(test_df.columns)
	self.test_index = test_df.Id
	train_df = train_df.astype(float)
	test_df = test_df.astype(float)
	#print(train_df.iloc[0].values)
	mapper = DataFrameMapper([
				 ([
		                   'Elevation',
				   'Aspect', 
				   'Slope',
       				   'Horizontal_Distance_To_Hydrology',
			           'Vertical_Distance_To_Hydrology',
       				   'Horizontal_Distance_To_Roadways',
				   'Hillshade_9am',
				   'Hillshade_Noon',
       				   'Hillshade_3pm',
				   'Horizontal_Distance_To_Fire_Points'
				  ], MinMaxScaler()
				),
					([
		         	   'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
				   'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
				   'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
				   'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
				   'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
				   'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
				   'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
				   'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
				   'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
				   'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
				   'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
				   'Soil_Type40'
					], None
				)
				 ]
                                )

	
	self.X_train = mapper.fit_transform(train_df)
	# print(X_train[0:2,:])

	self.y_train = train_df.Cover_Type.values
	# print(y_train[0:10])

	self.X_test = mapper.transform(test_df)
	# print(X_test[0:2,:])


    def runClassifier(self, classifyFunc, classifyArgs, gridsearchArgs=None, n_folds=10, exptNum=-1):
        """ Run classifier on the feature vectors """

        self.outRunDir = self.outDir + os.sep + 'expt_%d' % (exptNum) 

        self.outRunParamsDir = self.outRunDir + os.sep + 'params'
        self.outRunReportDir = self.outRunDir + os.sep + 'report'
        self.outRunFilesDir  = self.outRunDir + os.sep + 'files'
        utils.mkdir_p(self.outRunParamsDir)
        utils.mkdir_p(self.outRunReportDir)
        utils.mkdir_p(self.outRunFilesDir)
        self.CLFile = self.outRunReportDir + os.sep + 'cl.pkl'
        self.ParamsFile = self.outRunParamsDir + os.sep + 'params.pkl'
        self.visDir = self.outRunFilesDir + os.sep + 'classes'

        pinfo = {'algo': str(classifyFunc),
                 'params': classifyArgs,
                 'expt_num': exptNum}

        with open(self.ParamsFile, 'wb') as pf:
            pickle.dump(pinfo, pf)


	if exptNum == -1:  # use GridSearchCV
		clf = GridSearchCV(classifyFunc(**classifyArgs), gridsearchArgs,
				   scoring=None, 
				   loss_func=None,
				   score_func=None,
				   fit_params=None, 
				   n_jobs=4,  # n_jobs = -1
				   iid=True,
				   refit=True, 
				   cv=n_folds, 
				   verbose=1, 
				   pre_dispatch='2*n_jobs',
				   error_score='raise'
				  )
		# clf.fit(self.X_train, self.y_train)
		clf.fit(self.Y_bin, self.y_train)
		print('Grid Scores: \n', clf.grid_scores_)
		print('Best Estimator: \n', clf.best_estimator_)
		print('Best Score: ', clf.best_score_)
		print('Best Params: \n', clf.best_params_)
		print('Scorer: ', clf.scorer_)
		y_pred = clf.predict(self.Y_test_bin).astype(int)
                return y_pred
		# score = clf.score(self.X_test, y_pred)
		# clRes = {'score': score, 'clf': clf }
		# with open(self.CLFile, 'wb') as f2:
		#	pickle.dump(clRes, f2)
	else:
		# Cross Validation
		#t_start = time.time()
		N = self.X_train.shape[0]
		#kf = KFold(N, n_folds=10)
		#kf = LeaveOneOut(N)
		folds = n_folds
		kf = StratifiedKFold(self.y_train,folds)
		i=0
		sum_acc = 0
                Y = np.zeros(self.y_train.shape, dtype=int)
                Y_test = np.zeros((self.X_test.shape[0],1), dtype=int)
		for train,test in kf:
			X_train, X_test, y_train, y_test = self.X_train[train], self.X_train[test], self.y_train[train], self.y_train[test]
		# X_train, X_test, y_train, y_test = train_test_split(self.X_train,
		#						    self.y_train,
		#						    test_size=0.2,
		#						    random_state=self.randomSeed)
		# clf = LinearSVC()
			clf = classifyFunc(**classifyArgs)
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
                        Y[test] = y_pred
			score = clf.score(X_test, y_test)
			sum_acc = sum_acc + score
			CM = metrics.confusion_matrix(y_test, y_pred)

                # Fit and predict over complete training set now
		clf = classifyFunc(**classifyArgs)
		clf.fit(self.X_train, self.y_train)
                Y_test = clf.predict(self.X_test)
                
                if hasattr(clf, 'oob_score_'):
                    print('OOB Score: %f' % clf.oob_score_)

	#	# mAP = metrics.average_precision_score(y_test, y_pred)
	#	#t_end = time.time()
	#	# clRes = {'score': score, 'mAP':mAP, 'CM': CM, 'clf': clf }
	# 	clRes = {'score': score, 'CM': CM, 'clf': clf }
		print('Score (fold accuracy): %f' % score)
	#	# print('Score (mAP): %f' % mAP)
	#		with open(self.CLFile, 'wb') as f2:
	#			pickle.dump(clRes, f2)
	#	# print('Confusion Matrix:')
	#	# print(CM)
	#	#print('Time for runClassifier: %f sec' % (t_end-t_start))
			# self.showConfusionMatrix(i)
        #		i = i + 1
		avg_acc = sum_acc / float(folds)
		print('Average Accuracy %f' % avg_acc)
                return (Y, Y_test)


#		clf = classifyFunc(**classifyArgs)
#		clf.fit(self.X_train, self.y_train)

#	if hasattr(clf, 'oob_score_'):
#		print('OOB Score: %f' % clf.oob_score_)
#	y_test_pred = clf.predict(self.X_test)
#	y_test_pred = y_test_pred.astype(int)
#
#	# print(y_pred)
#        return y_test_pred


    def showConfusionMatrix(self,i):
        with open(self.CLFile, 'rb') as f:
            clRes = pickle.load(f)
            print('Score (mean accuracy): %f' % clRes['score'])
            # print('Score (mAP): %f' % clRes['mAP'])
            plt.matshow(clRes['CM'])
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # plt.show()
            #plt.savefig('cm_%d.png' % (i))

    def runAlgo(self, algo, fixedParamsDict, variableParamsDict, cv=False, n_folds=10):

	y_pred = []


	if cv:
		# Select best parameters by grid search and cross validation
		y_pred = self.runClassifier(algo, fixedParamsDict,
						 gridsearchArgs=variableParamsDict,
						 n_folds=n_folds, 
						 exptNum=-1)

		outputDir = self.outDir + os.sep
		self.save_sub(outputDir, y_pred)
	else:
		A = variableParamsDict
		X = list(itertools.product(*A.values()))
		D = [dict(zip(A.keys(), x)) for x in X]
                Y = np.zeros((self.y_train.shape[0], len(D)))
                Y_test = np.zeros((self.X_test.shape[0], len(D)))
		exptCtr = 0
		for d in D:
		    params = {}
		    params = d.copy()
		    params.update(fixedParamsDict)
		    print(params)
		    y_train_pred, y_test_pred = self.runClassifier(algo, params, exptNum=exptCtr, n_folds=n_folds)
		    y_pred.append(y_train_pred)
                    Y[:,exptCtr] = y_train_pred
                    Y_test[:,exptCtr] = y_test_pred
		    # outputDir = self.outDir + os.sep + 'expt_%d' % (exptCtr)
		    # self.save_sub(outputDir, y_test_pred)
		    exptCtr = exptCtr + 1
		# y_pred = self.runClassifier(algo, params, exptNum=n_folds, n_folds=n_folds)	
                return (Y, Y_test) 
            
	return y_pred

    def save_sub(self, outputDir, y_test_pred):
	
	pred_df = pd.DataFrame(y_test_pred, columns=['Cover_Type'])
	print(pred_df.head())
	out_df = pd.concat([self.test_index, pred_df], axis=1)
	print(out_df.head())
	fname = outputDir + os.sep + self.runinfo['git_rev'][0:8] + ".csv"
	print(fname)
	utils.mkdir_p(outputDir)
	out_df.to_csv(fname, index=False)

    def classify(self):

        ## 1) Random Forest
        rfVarParams = {
                       'n_estimators':[100, 1000],
                       'criterion':['gini', 'entropy'],
                       'max_depth':[None, 3, 10, 100]
                       }

        rfFixedParams = {
                         'min_samples_split':2,
                         'min_samples_leaf':1,
                         'min_weight_fraction_leaf':0.0,
                         'max_features':'auto',
                         'max_leaf_nodes':None,
                         'bootstrap':True,
                         'oob_score':True,
                         'n_jobs':4,
                         'random_state':self.randomSeed,
                         'verbose':1,
                         'warm_start':False,
                         'class_weight':'auto'
                        }

        ## 2) XGBoost
        xgbVarParams = {}
        xgbFixedParams = {'silent':0}

        ## 3) Nearest Neighbour
        nnVarParams = {
                       'n_neighbors':[5, 10, 15],
                       'algorithm':['auto']#,'ball_tree','kd_tree','brute']
                      }

        nnFixedParams = {
                         'weights':'uniform',
                         'leaf_size':30, 
                         'p':2,
                         'metric':'minkowski', 
                         'metric_params':None
                        }

        ## 4) LinearSVC
        linearSVCVarParams = {
                              'penalty': ['l1'],
                              'loss': ['squared_hinge'],
                              'C': [0.25, 0.5, 0.75, 1.0],
                              }
        linearSVCFixedParams = {
                                'dual':False,
                                'tol':0.0001,
                                'multi_class':'ovr',
                                'fit_intercept':True,
                                'intercept_scaling':1,
                                'class_weight':'auto',
                                'verbose':1,
                                'random_state':self.randomSeed,
                                'max_iter':1e6
                               }

        ## 5) SVM (kernel)
        SVCVarParams = {'C':[0.25, 0.5, 0.75, 1.0]}
        SVCFixedParams = { 'kernel':'rbf',
                           'degree':3,
                           'gamma':0.0,
                           'coef0':0.0,
                           'shrinking':True,
                           'probability':False, 
                           'tol':0.001,
                           'cache_size':200,
                           'class_weight':'auto',
                           'verbose':False, 
                           'max_iter':-1,
                           'random_state':self.randomSeed
                         }

        ## 6) ADA Boost
        adaVarParams = { 
                        'base_estimator':[None],
                        'n_estimators':[50],
                        'learning_rate':[1.0]
                       }
        adaFixedParams = {
                          'algorithm':'SAMME.R',
                          'random_state':self.randomSeed
                         }   

        ## 7) GradientBoostClassifier
        gbcVarParams = {'loss':['deviance', 'exponential'], 
                        'learning_rate':[0.01, 0.1, 0.5, 1.0],
                        'n_estimators':[100, 1000],
                        'max_depth':[3, 30, 300],
                        'subsample':[0.25, 0.5, 0.75, 1.0],
                       }
        gbcFixedParams = {  
                          'min_samples_split':2,
                          'min_samples_leaf':1,
                          'min_weight_fraction_leaf':0.0, 
                          'init':None,
                          'random_state': self.randomSeed,
                          'max_features':None,
                          'verbose':1,
                          'max_leaf_nodes':None,
                          'warm_start':False
                         }

        ## 8) naive-bayes
        nbVarParams = { }
        nbFixedParams = { }

        ## 9) Logistic Regression
        logisticVarParams = { 'penalty': ['l1', 'l2'],
                              'C': [0.25, 0.5, 0.75, 1.0]
                            }
        logisticFixedParams = {
                               'dual':False,
                               'tol':0.0001,
                               'fit_intercept':True,
                               'intercept_scaling':1,
                               'class_weight':None,
                               'random_state':self.randomSeed,
                               'solver':'liblinear',
                               'max_iter':1000000,
                               'multi_class':'ovr',
                               'verbose':1
                              }

        classifier = [
                      ('Random-Forests', RandomForestClassifier, rfFixedParams, rfVarParams),
                      ('GradientBoost', GradientBoostingClassifier, gbcFixedParams, gbcVarParams),
                      ('LibLinear', LinearSVC, linearSVCFixedParams, linearSVCVarParams),
                      ('LibSVM', SVC, SVCFixedParams, SVCVarParams),
                      ('NearestNeighbour', KNeighborsClassifier, nnFixedParams, nnVarParams),
                     ]

        utils.mkdir_p(self.outDir)
        self.readDataset()
       
        y_pred = []
        Y = np.empty((self.X_train.shape[0], 0))
        Y_test = np.empty((self.X_test.shape[0], 0))
        for (clname, cl, fixedParams, varParams) in classifier:
            yp, yp_test = self.runAlgo(cl, fixedParams, varParams, cv=False)
            Y = np.hstack((Y, yp))
            Y_test = np.hstack((Y_test, yp_test))
            y_pred.append(yp)
            print(clname, len(yp))

        ohe = OneHotEncoder(n_values=7, dtype=int)
        self.Y_bin = ohe.fit_transform(Y-1).astype(float)
        self.Y_test_bin = ohe.fit_transform(Y_test-1).astype(float)
        pred = self.runAlgo(LogisticRegression, logisticFixedParams, logisticVarParams, cv=True)
