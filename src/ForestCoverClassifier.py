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
import matplotlib.pyplot as plt
import itertools
import sys
import pandas as pd
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.preprocessing import MinMaxScaler
#from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
#from sklearn.cross_validation import LeaveOneOut

class ForestCoverClassifier:
    """Forest Cover Classifer"""

    def __init__(self, runinfo, settings='../settings.ini'):
        """Initialize object"""

        self.runinfo = runinfo
        print('Numpy:', np.__version__)
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


    def runClassifier(self, classifyFunc, classifyArgs, exptNum, patches=True):
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


	# Cross Validation
	#t_start = time.time()
#	N = self.X_train.shape[0]
#	#kf = KFold(N, n_folds=10)
#	#kf = LeaveOneOut(N)
#	folds = 10
#	kf = StratifiedKFold(self.y_train,folds)
#	i=0
#	sum_acc = 0
#	for train,test in kf:
#		X_train, X_test, y_train, y_test = self.X_train[train], self.X_train[test], self.y_train[train], self.y_train[test]
#	# X_train, X_test, y_train, y_test = train_test_split(self.X_train,
#	#						    self.y_train,
#	#						    test_size=0.2,
#	#						    random_state=self.randomSeed)
#	# clf = LinearSVC()
#		clf = classifyFunc(**classifyArgs)
#		clf.fit(X_train, y_train)
#		y_pred = clf.predict(X_test)
#		score = clf.score(X_test, y_test)
#		sum_acc = sum_acc + score
#		CM = metrics.confusion_matrix(y_test, y_pred)
#	# mAP = metrics.average_precision_score(y_test, y_pred)
#	#t_end = time.time()
#	# clRes = {'score': score, 'mAP':mAP, 'CM': CM, 'clf': clf }
#		clRes = {'score': score, 'CM': CM, 'clf': clf }
#		print('Score (mean accuracy): %f' % score)
#	# print('Score (mAP): %f' % mAP)
#		with open(self.CLFile, 'wb') as f2:
#			pickle.dump(clRes, f2)
#	# print('Confusion Matrix:')
#	# print(CM)
#	#print('Time for runClassifier: %f sec' % (t_end-t_start))
#		self.showConfusionMatrix(i)
#		i = i + 1
#	avg_acc = sum_acc / float(folds)
#	print('Average Accuracy %f' % avg_acc)


	clf = classifyFunc(**classifyArgs)
	clf.fit(self.X_train, self.y_train)
	self.y_pred = clf.predict(self.X_test)
	self.y_pred = self.y_pred.astype(int)

	# print(y_pred)


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
            plt.savefig('cm_%d.png' % (i))

    def runAlgo(self, algo, fixedParamsDict, variableParamsDict, patches=True):

        utils.mkdir_p(self.outDir)

        self.readDataset()

        A = variableParamsDict
        X = list(itertools.product(*A.values()))
        D = [dict(zip(A.keys(), x)) for x in X]

        exptCtr = 0
        for d in D:
            params = {}
            params = d.copy()
            params.update(fixedParamsDict)
            print(params)
            self.runClassifier(algo, params, exptCtr)
            exptCtr = exptCtr + 1

    def save_sub(self):
	
	pred_df = pd.DataFrame(self.y_pred, columns=['Cover_Type'])
	# print(pred_df.head())
	out_df = pd.concat([self.test_index, pred_df], axis=1)
	# print(out_df.head())
	fname = self.runinfo['git_rev'][0:8] + ".csv"
	print(fname)
	out_df.to_csv(fname, index=False)
		

    def classify(self):

        # Classification
        ## 1) LinearSVC
        linearSVCVarParams = {'penalty': ['l2']}
        linearSVCFixedParams = {'loss':'squared_hinge',
                                'dual':True,
                                'tol':0.0001,
                                'C':1.0,
                                'multi_class':'ovr',
                                'fit_intercept':True,
                                'intercept_scaling':1,
                                'class_weight':'auto',
                                'verbose':1,
                                'random_state':self.randomSeed,
                                'max_iter':1e6
                               }
        self.runAlgo(LinearSVC, linearSVCFixedParams, linearSVCVarParams)
	

#        SVCVarParams = {'C':[1.0]}
#        SVCFixedParams = { 'kernel':'rbf',
#                           'degree':3,
#			   'gamma':0.0,
#			   'coef0':0.0,
#			   'shrinking':True,
#			   'probability':False, 
#			   'tol':0.001,
#			   'cache_size':200,
#			   'class_weight':'auto',
#			   'verbose':False, 
#			   'max_iter':-1,
#			   'random_state':self.randomSeed
#			}
#
#        self.runAlgo(SVC, SVCFixedParams, SVCVarParams)



	self.save_sub()

