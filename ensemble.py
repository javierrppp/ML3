import pickle
import feature
from PIL import Image
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.alphas = []
        self.weak_classifier_list = []
        self.weight = None

    def is_good_enough(self):
        '''Optional'''
        pass
    
    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        if self.weight is None:
            self.weight = np.zeros(X.shape[0])
            for i in range(self.weight.shape[0]):
                self.weight[i] = 1/self.weight.shape[0]
        self.weak_classifier = DecisionTreeClassifier(criterion='gini')
        self.weak_classifier.fit(X, y,sample_weight=self.weight)
        return self.weak_classifier
       


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        h= np.zeros(X.shape[0])
        for i in range(len(self.weak_classifier_list)):
            h = h+self.alphas[i]*self.weak_classifier_list[i].predict(X)
        for i in range(h.shape[0]):
            if h[i]<=0:
                h[i]=-1
            else:
                h[i]=1
        return h

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        xTest = self.weak_classifier.predict(X)
        return xTest

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
