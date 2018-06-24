import numpy as np
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import BallTree

class DR:
    def __init__(self, X, K):
        self.X = np.array(X)
        self.N = self.X.shape[0]
        self.D = np.zeros((self.N, self.N))
        self.W = np.zeros((self.N, self.N))
        self.K = K

    def construct_similarity_graph(self):
        '''
        # cosine similarity
        self.W = cosine_similarity(self.X)
        print(self.W.shape)
        for i in range(self.N):
            for j in range(self.N):
                if self.W[i][j] < 0:
                    self.W[i][j] = 0
        print(self.W)
        '''
        '''
        # rbf kernel
        self.W = rbf_kernel(self.X)
        print(self.W)
        '''
        # k-nearest neighbor graph
        tree = BallTree(self.X, metric="manhattan")
        for i in range(self.N):
            dist, index = tree.query([self.X[i]], k=5)
            max_dist = max(dist[0])
            for indice, j in enumerate(index[0]):
                distance = dist[0][indice]
                #print(distance)
                #similarity = 1 / (1 + distance)
                similarity = 1 - (distance / max_dist)
                #similarity = distance
                self.W[i][j] = similarity
                self.W[j][i] = similarity

    def construct_degree_matrix(self):
        for i in range(self.N):
            self.D[i][i] = np.sum(self.W[i])
        print(self.D)

    def construct_laplacian_matrix(self):
        self.L = self.D - self.W

    def eigenmap(self):
        eigenvalues, eigenvectors = sparse.linalg.eigs(self.L, k=self.K, M=self.D, which="SM")
        #print(eigenvectors.shape)
        #print(eigenvectors)
        eig_seq = np.argsort(eigenvalues)
        #print(eigenvalues[eig_seq])
        eig_seq_indice = eig_seq[0:self.K]
        new_eig_vec = eigenvectors[:,eig_seq_indice]
        print(new_eig_vec.shape)
        return new_eig_vec

    def analyze(self):
        self.construct_similarity_graph()
        self.construct_degree_matrix()
        self.construct_laplacian_matrix()
        return self.eigenmap()