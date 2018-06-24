### 彭显敬 2016141462055 MNIST Clustering Analysis
### all the scores are based on AMI
### number of clusters is fixed at 10
### all the experiments were done in 
- MacBook Air 1.8 GHz Intel Core i5
### the source code and this report are available at
- https://github.com/Stepphonwol/MNIST-Clustering-Analysis
### This report could be divided into three parts
- Experiment and analysis of k-Means
- Experiment and analysis of Spectral Clustering
- Conclusion, problems and expectation
## k-Means
### basic initialization
$$\sigma*np.random.rand(self.k)+\mu$$
where $\sigma$ is the standard deviation of the data set $\chi$, $\mu$ is the mean of $\chi$.
#### N = 1000 iterating 100 times
| | BASIC | sklearn |
| - | :-: | :-: |
| | 0.4856 54.36s | 0.5016 0.9526 |
| | 0.5272 60.02s | 0.5211 0.8851 |
| | 0.4977 59.34s | 0.5157 0.9162 |
| | 0.4627 57.68s | 0.5293 0.8566 |
| | 0.5109 57.20s | 0.5300 0.8867 |
| average | **0.4968** 57.72s | **0.5195** 0.8994s |
| variance | **0.0005** 3.89 | **0.0001** 0.001 |
#### N = 10000 iterating 100 times
| | BASIC | sklearn |
| - | :-: | :-: |
| | 0.4830 578.11s | 0.5130 10.91s |
| | 0.4843 603.76s | 0.5130 10.05s |
| | 0.4863 602.46s | 0.5064 11.68s |
| | 0.5024 604.45s | 0.4928 9.84s |
| | 0.4771 618.19s | 0.5018 9.35s |
| average | **0.4866** 601.39s | **0.5054** 10.37s  |
| variance | **7.16e-5** 168.06 | **5.76e-5** 0.6862  |
#### conclusion
- Basic k-Means is not sensitive to the size of the data set. One possible reason is that the initialization method chooses the initial centroids **too uniformly** which fail to depict the exact distribution of $\chi$.
- It's too difficult to set the threshold of k-Mean's convergence. Thus, the times of iterating is strictly controlled.
### ++ initialization
- Choose one centroid randomly from $\chi$
- For each $x$ in $\chi$, calculate the distance $D(x)$ from $x$ to its nearest neighbors in all the centroids choosen
- draw a centroid $x'$ in $\chi$ with the probability $\frac{D(x')}{\sum D(x)}$
- Repeat the two steps above until k centers have been chosen
```PYTHON
        for i in range(1, self.K):
            dist_list = np.array([min(np.linalg.norm(c - x) for c in self.centroids) for x in self.X])
            # calculate and record the nearest distance
            # normalization
            dist_list = dist_list / np.sum(dist_list)
            # construct probability distribution
            prob_distribution = dist_list.cumsum()
            # simulate the probability of choosing a centroid
            p_centroid = np.random.random()
            for j, p in enumerate(prob_distribution):
                if p_centroid < p:
                    self.centroids[i] = self.X[j]
                    break
```
#### N=1000 iterating 100 times
| | ++ | sklearn |
| - | :-: | :-: |
| | 0.4891 36.82s | 0.5016 0.9526 |
| | 0.4914 39.82s | 0.5211 0.8851 |
| | 0.4513 42.62s | 0.5157 0.9162 |
| | 0.4752 41.39s | 0.5293 0.8566 |
| | 0.5188 46.66s | 0.5300 0.8867 |
| average | **0.4852** 41.46s | **0.5195** 0.8994s |
| variance | **0.0005** 10.52 | **0.0001** 0.001 |
#### conclusion
- No major advantage compared with basic k-Means, it is even worse. It seems that the limitation of k-Means has been reached. Note that the level of redundancy must be high in $\chi$, thus, maybe spectral clustering method which reduces the original dimension while clustering, may be better.
## Spectral Clustering
### Fully Connected Graph
#### N=1000 k-Means++ iterating 100 times
| | RBF Kernel | Cosine Similarity |
| - | :-: | :-: |
| | 0.4046 12.78s | 0.4742 14.5s |
| | 0.4553 13.21s | 0.4572 15.58s |
| | 0.4361 14.73s | 0.4669 15.27s |
| | 0.4308 16.00s | 0.4300 15.83s |
| | 0.4918 13.53s | 0.4972 16.9s |
| | 0.4645 11.99s | 0.4933 15.68 |
| | 0.4603 13.32s | 0.4647 16.80s |
| | 0.4949 12.86s | 0.5083 17.13s |
| | 0.4412 13.48s | 0.5027 17.40s |
| | 0.4669 13.44s | 0.5169 17.00s |
| average | **0.4546** 13.43s | **0.4811** 16.21s |
| varaince | **0.0007** 1.20 | **0.0007** 0.84 |
#### conclusion
- Both RBF kernel and cosine similarity fail to depict the relationship between the data points properly. (why)
### KNN Graph
#### $k$ : number of neighbors when building the KNN graph
#### first attempt
- Choose k as the number of clusters, 10
- In this very first experiment of spectral clustering, I accidentally set the similarity as the euclidean distance, as the code followed. In the rest of the experiments, the similarity was set as $\frac{1}{1 + distance}$. A discussion between these two metrics is also included in this report. 
```PYTHON
        tree = BallTree(self.X)
        for i in range(self.N):
            dist, index = tree.query([self.X[i]], k=8)
            for indice, j in enumerate(index[0]):
                similarity = dist[0][indice]
                self.W[i][j] = similarity
                self.W[j][i] = similarity
```
- Note that in the previous experiments, we could not tell whether k-Means++ is better than k-Means. Thus, we have to decide one now.
##### a comparison between k-Means++ and basic k-Means
| | ++ | BASIC |
| - | :-: | :-: |
| | 0.5211 10.75s | 0.4807 10.28s |
| | 0.5319 11.09s | 0.4290 9.99s |
| | 0.4982 11.02s | 0.4714 10.53s |
| | 0.5116 12.06s | 0.5025 10.57s |
| | 0.4577 11.06s | 0.4545 10.36s |
| | 0.5228 11.20s | 0.4102 10.80s |
| | 0.5383 12.11s | 0.4612 10.48s |
| | 0.4795 11.20s | 0.4460 9.95s |
| | 0.4434 11.62s | 0.4478 10.30s |
| | 0.5077 11.67s | 0.5170 9.96s |
| average | **0.5012** | **0.4620** |
| variance | **0.0003** | **0.0009** |
##### conclusion
- When $\chi$ is reduced to 10 dimensions by spectral analysis during which the level of redundancy declines, k-Means++ shows its superiority over basic k-Means.
- However, spectral clustering does not overperform k-Means++ a lot when $k$ is set as 10, thus, we need to tune the parameter $k$.
- At the beginning of the experiment, I was not aware of the use of sparse matrix, and adopt numpy to do all the inversion and eigenvalue decomposition, which took a lot of time. Thus, all the tests were all first based on 1000 samples. **And I started to think maybe there is some sort of connection between the small samples and big samples. More specifically, a good $k$ on small samples may still be a good $k$ on big samples.** Based on assumptions and targets above, the following experiments were conducted.
#### N=1000 k-Means++ iterating 100 times (using numpy)
| | $k=3$ | $k=4$ | $k=5$ | $k=6$ | $k=7$ | $k=8$ | $k=9$ | $k=10$ | $k=11$ |
| - | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| | 0.4163 11.62s | 0.5114 14.11s | 0.4453 14.92s | 0.4916 11.25s | 0.5442 11.84s | 0.5014 11.45s | 0.5227 10.99s | 0.5034 15.6s | 0.5144 12.04s |
| | 0.3653 11.11s | 0.5332 14.36s | 0.4720 14.81s | 0.4886 11.31s | 0.5421 11.84s | 0.5109 10.80s | 0.5060 11.02s | 0.5208 15.82s | 0.5069 10.98s |
| | 0.4175 11.98s | 0.5200 15.01s | 0.5315 15.04s | 0.5395 12.01s | 0.5256 11.14s | 0.5286 11.21s | 0.5133 12.34s | 0.5236 15.67s | 0.5066 10.84s |
| | 0.3382 11.40s | 0.4697 15.24s | 0.5083 14.90s | 0.5015 10.99s | 0.5098 11.91s | 0.4850 11.42s | 0.5106 11.49s | 0.5216 15.61s | 0.5020 10.79s |
| | 0.3952 11.64s | 0.5127 15.16s | 0.4808 14.89s | 0.5415 11.14s | 0.5570 12.11s | 0.5419 11.36s | 0.5023 11.29s | 0.5040 15.88s | 0.4782 11.94s |
| | 0.4167 10.80s | 0.4990 14.16s | 0.5388 14.56s | 0.5107 11.04s | 0.5263 11.41s | 0.4872 10.98s | 0.5694 11.38s | 0.5187 15.40s | 0.5400 12.21s |
| | 0.3713 13.05s | 0.5328 15.19s | 0.5240 15.34s | 0.5047 11.40s | 0.5531 11.70s | 0.5157 11.55s | 0.5004 10.94s | 0.5376 15.78s | 0.5173 11.44s |
| | 0.3415 10.82s | 0.4737 14.83s | 0.5104 14.56s | 0.4907 11.09s | 0.4912 11.44s | 0.4665 11.70s | 0.5122 11.32s | 0.4928 16.12s | 0.5043 10.92s |
| | 0.3640 11.10s | 0.5162 14.87s | 0.5481 15.36s | 0.5164 11.32s | 0.5920 11.18s | 0.5267 10.80s | 0.5426 11.57s | 0.5353 16.47s | 0.5051 10.84s |
| | 0.4018 11.14s | 0.4751 14.72s | 0.4187 14.81s | 0.5183 11.42s | 0.5267 11.06s | 0.5502 11.51s | 0.5025 11.42s | 0.5363 15.39s | 0.4729 10.82s |
| average | **0.3828** | **0.5044** | **0.4978** | **0.5103** | **0.5368** | **0.5114** | **0.5182** | **0.5194** | **0.5048** |
| variance | **0.0008** | **0.0005** | **0.0016** | **0.0003** | **0.0007** | **0.0006** | **0.0004** | **0.0002** | **0.0003** |
#### N=10000 k-Means++ iterating 100 times (using scipy.sparse)
| | $k=3$ | $k=4$ | $k=5$ | $k=6$ | $k=7$ | $k=8$ | $k=9$ | $k=10$ | $k=11$ |
| - | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| | | 0.5846 527.62s | 0.7208 498.25s | 0.6651 574.89s | 0.6898 544.25s | 0.6283 523.80s | 0.6404 488.69s | 0.5907 426.88s | 0.6054 461.08s |
| | | 0.7173 541.56s | 0.7120 498.49s | 0.7147 584.38s | 0.6867 521.46s | 0.6544 540.94s | 0.6587 473.90s | 0.6080 432.41s | 0.6188 441.97s |
| | | 0.6862 548.47s | 0.7366 459.32s | 0.6965 548.41s | 0.6457 537.69s | 0.6523 505.23s | 0.6550 460.43s | 0.6772 427.22s | 0.6879 420.79s |
| | | 0.6797 527.20s | 0.6645 490.62s | 0.6183 538.51s | 0.6909 511.54s | 0.6603 537.70s | 0.6811 428.18s | 0.6589 418.57s | 0.6749 404.52s |
| | | 0.7042 583.33s | 0.6856 465.64s | 0.6370 546.12s | 0.6559 516.53s | 0.6406 588.47s | 0.6718 484.03s | 0.6336 419.43s | 0.5594 477.25s
| average | | **0.6744** | **0.7039** | **0.6663** | **0.6738** | **0.6472** | **0.6614** | **0.6337** | **0.6293** |
| variance | | **0.0021** | **0.0007** | **0.0013** | **0.0003** | **0.0001** | **0.0002** | **0.0010** | **0.0022** |
##### conclusion
- The performance of spectral clustering with KNN similarity graph based on the inverse of euclidean distance varied from small data set to big data set. It turns out that a good $k$ on small samples is not necessarilly a good $k$ on big samples.
    - When $N=1000$, $k=7$ is clearly the best one(**0.5368**) which outperformed other $k$ a lot.
    - When $N=10000$, $k=5$ is clearly the best one(**0.7039**) which outperformed other $k$ a lot.
    - Apparently, the reason is that for different sizes of the data set, different similarity graphs are constructed, thus, different graph cut are being made.
- Also note that when $N=10000$, the overall variance of AMI is bigger than $N=1000$. And for some values of $k$ far away from the $5$, they also did a good job.
#### About the Distance Metric
- In the former experiments, I chose euclidean distance as the metric. However, it's common knowledge that in high dimensions (in this case, 784 dimensions), it's hard to tell the close neighbors from the distant neighbors. In other words, data points become uniformly distant from each other in high dimensions. To solve this problem, I decided to try on Manhattan distance metric.
- Before entering the comparison between euclidean distance and manhattan distance, let's first compare two different definitions of similarity:
    - set the distance as similarity directly
    - set $\frac{1}{1+distance}$ as the similarity
    (note that the first definition is not a strict one, because for two identical data points, their similarity is not one but zero.)

| average AMI and variance | $distance$ | $\frac{1}{1+distance}$ |
| - | :-: | :-: |
| $k=4$ | **0.6615** 0.0114 | **0.6744** 0.0021 |
| $k=5$ | **0.6355** 0.0018 | **0.7039** 0.0007 |
| $k=6$ | **0.6528** 0.0011 | **0.6663** 0.0013 |
| $k=7$ | **0.6740** 0.0018 | **0.6738** 0.0003 |
| $k=8$ | **0.6091** 2.04e-5 | **0.6472** 0.0001 |
- Apparently, the second definition is better than the first one.
##### a comparison between Manhattan distance and Euclidean distance
based on similarity metric : $\frac{1}{1+distance}$
| average AMI and variance | $L1$ | $L2$ |
| - | :-: | :-: |
| $k=4$ | **0.6601** 0.0009 | **0.6744** 0.0021 |
| $k=5$ | **0.6474** 0.0012 | **0.7039** 0.0007 |
| $k=6$ | **0.6410** 0.0007 | **0.6663** 0.0013 |
| $k=7$ | **0.6352** 0.0006 | **0.6738** 0.0003 |
| $k=8$ | **0.6242** 0.0021 | **0.6472** 0.0001 |
##### conclusion
- Euclidean distance is better than manhattan distance, when based on the similarity definition $\frac{1}{1+distance}$.(as for why, I don't know)
#### Final Attempt
- Another definition of similarity : 
$$ 1-\frac{distance}{max(distance)}$$
set $k=5$ (the tuned parameter)

| | L1 | L2 |
| - | :-: | :-: |
| 1 | **0.7024** 810.94s | **0.7285** 965.14s |
| 2 | **0.6944** 796.54s | **0.7225** 1297.29s |
| 3 | **0.6314** 820.54s | **0.7346** 1045.46s |
| 4 | **0.5601** 831.28s | **0.6750** 965.65s |
| 5 | **0.6972** 789.57s | **0.7277** 984.21s |
| average & variance | **0.6571** 0.0030 | **0.7177** 0.0005 |
##### conclusion
- The new definition didn't lead to a substantial improvement.
## Final Conclusion 
### number in the bracket indicates average AMI score and variance
- This report began at basic k-Means(**0.4866**, 7.16e-5), to k-Means++(**0.4852**, 0.0005), trying to address the problem of initialization. No big improvement here.
- Then Spectral Clustering is introduced. First, a fully-connected graph based on RBF kernel(**0.4546**, 0.0007) and cosine similarity(**0.4811**, 0.0007) is used, which generates even poorer results than k-Means.
- Then Spectral Clustering based on KNN graph and euclidean distance is adopted. Due to the limitation of computational sources, a small sampe $N=1000$ was first used to tune the parameter $k$, which mounted at $k=7$(**0.5386**, 0.0007). 
- After using scipy.sparse, the calculation was significantly boosted. Thus, $N=10000$ was tested. The parameter $k$ was tuned at $5$(**0.7039**, 0.0007)
- Different definitions of distance-based similarity and differnet distances(L1 and L2) were tested. The winner is L2 norm with the similarity definition : $1-\frac{distance}{max(distance)}$ (**0.7177**, 0.0005)
## Problems and Expectation
- Due to the limitation of computational sources, the experiments are not enough.
- k-Means++ didn't work as I had expected.
- Too much time was spent on tuning the parameter and determing the optimum distance metric and similarity definition. The method of metric learning may be introduced.


