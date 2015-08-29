from __future__ import print_function
import caffe
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
import matplotlib.pyplot as plt
import itertools
import sys


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
        self.dsetLoc = self.config.get('dataset', 'dset_loc')
        self.scratchLoc = self.config.get('system', 'scratch_loc')
        self.outDir = self.config.get('system', 'out_dir')
        self.FVFile = self.outDir + os.sep + self.config.get('system', 'fv_file')
        self.FVFileCpp = self.outDir + os.sep + self.config.get('system', 'fv_file_cpp')
        self.FVDim = int(self.config.get('caffe', 'fv_dim'))
        self.FVLayerName = self.config.get('caffe', 'feature_layer')
        print(self.dsetName)
        print(self.dsetLoc)
        print(self.outDir)
        print(self.FVLayerName)
        np.set_printoptions(threshold=np.nan)   # Print complete array
        self.clusters = {}

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

        with open(self.FVFile, 'rb') as f:
            t_start = time.time()
            self.db = pickle.load(f)
            y = np.array(self.db['labels'])
            print("#Features:", self.db['numImages'])
            N = self.db['numImages']
            D = int(self.FVDim)
            X = np.zeros((N, D))
            for i in range(N):
                # print(self.db['FV'][i])
                X[i, :] = self.db['FV'][i].ravel()
                # pdb.set_trace()
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                    test_size=0.2, random_state=self.randomSeed)
            # clf = LinearSVC()
            clf = classifyFunc(**classifyArgs)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = clf.score(X_test, y_test)
            CM = metrics.confusion_matrix(y_test, y_pred)
            mAP = metrics.average_precison_score(y_test, y_pred)
            t_end = time.time()
            clRes = {'N': N, 'D': D, 'score': score, 'mAP':mAP, 'CM': CM, 'clf': clf }
            print('Score (mean accuracy): %f' % score)
            print('Score (mAP): %f' % mAP)
            with open(self.CLFile, 'wb') as f2:
                pickle.dump(clRes, f2)
            # print('Confusion Matrix:')
            # print(CM)
            print('Time for runClassifier: %f sec' % (t_end-t_start))
            self.showConfusionMatrix()

    def showConfusionMatrix(self):
        with open(self.CLFile, 'rb') as f:
            clRes = pickle.load(f)
            print('Score (mean accuracy): %f' % clRes['score'])
            print('Score (mAP): %f' % clRes['mAP'])
            plt.matshow(clRes['CM'])
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # plt.show()
            plt.savefig('cm.png')

    def runAlgo(self, algoType, algo, fixedParamsDict, variableParamsDict, patches=True):

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
                                'class_weight':None,
                                'verbose':1,
                                'random_state':self.randomSeed,
                                'max_iter':1e6
                               }
        self.runAlgo(self.runClassifier, LinearSVC, linearSVCFixedParams, linearSVCVarParams)

