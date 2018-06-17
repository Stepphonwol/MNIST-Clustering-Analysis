import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from extractor import DR
from sklearn import metrics
from sklearn.cluster import KMeans

class Clusterer:
    def __init__(self):
        raw_X, self.groundtruth = self.fetch_input()
        # k =10 due to MNIST
        #self.X = DR(raw_X, 10).analyze()
        self.X = raw_X
        self.N = self.X.shape[0]
        self.k = self.X.shape[1]
        self.K = 10
        start = time.time()
        clusterer = KMeans(n_clusters=self.K5)
        self.labels = clusterer.fit_predict(self.X)
        self.evaluate()
        end = time.time()
        print(end-start)
        '''
        self.labels = np.zeros(self.N)
        print(self.labels.shape)
        # k-Means process
        start = time.time()
        #self.preprocessing()
        self.input_initialization()
        self.EM()
        self.evaluate()
        end = time.time()
        print(end-start)
        '''

    def fetch_input(self):
        mat = sio.loadmat("mnist.mat")
        data = mat['data']
        label = mat['label'].flatten()
        print(data.shape)
        print(label.shape)
        return data, label

    def preprocessing(self):
        for i in range(self.N):
            for j in range(self.k):
                if self.X[i][j] < 0.01:
                    self.X[i][j] = 0
                else:
                    self.X[i][j] = 2

    def input_initialization(self):
        # basic initialization
        self.centroids = []
        global_means = np.mean(self.X)
        global_variance = np.std(self.X)
        for i in range(self.K):
            self.centroids.append(global_variance * np.random.randn(self.k) + global_means)
        # k-means++ initialization

    def EM(self):
        # clusters : the set of indexes for each cluster
        clusters = []
        for i in range(self.K):
            cluster = []
            clusters.append(cluster)
        #old_labels = np.zeros(self.N)
        max_iter = 100
        while max_iter > 0:
            print(max_iter)
            #old_labels = self.labels
            # update the label for each data point
            for i in range(self.N):
                min_dist = 0
                label = 0
                for j in range(self.K):
                    # calculate the distance from the data point to each centroid
                    dist = np.linalg.norm(self.X[i] - self.centroids[j])
                    if j == 0:
                        min_dist = dist
                    if dist < min_dist:
                        min_dist = dist
                        label = j
                self.labels[i] = label
                clusters[label].append(i)
            # update the centroid of each cluster
            for i in range(self.K):
                self.centroids[i] = np.mean(self.X[clusters[i]], axis=0)
                #print(self.centroids[i])
            max_iter -= 1

    def evaluate(self):
        score = metrics.adjusted_mutual_info_score(self.groundtruth, self.labels)
        print(score)


if __name__ == "__main__":
    c = Clusterer()