from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import wkpp as wkpp 
import numpy as np
import random


# Real data input
dataset = fetch_kddcup99()								# Fetch kddcup99 
data = dataset.data										# Load data
data = np.delete(data,[0,1,2,3],1) 						# Preprocess
data = data.astype(float)								# Preprocess
data = StandardScaler().fit_transform(data)				# Preprocess

n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)


def D2(data,k):											# D2-Sampling function.


	#---------- code for Algo-1----------#

	# Initialize the centers with the first randomly selected point
    centers = [data[random.randint(0, len(data) - 1)]]

    # Sample subsequent centers
    for _ in range(k - 1):
        # Calculate squared distances from each point to the nearest center
        distances = np.array([min(cdist([x], centers, 'sqeuclidean')[0]) for x in data])

        # Calculate probabilities proportional to the squared distances
        probs = distances / distances.sum()

        # Choose the next center based on the calculated probabilities
        next_center = random.choices(data, probs)[0]

        # Add the next center to the list of centers
        centers.append(next_center)

    return np.array(centers)

centers = D2(data,k)									# Call D2-Sampling (D2())

def Sampling(data, k, centers, Sample_size):  # Coreset construction function.

	#---------- code for Algo-1----------#
	
    # Compute squared distances from each point to each center
    distances = cdist(data, centers, 'sqeuclidean')

    # Assign weights based on the closest center
    weights = np.min(distances, axis=1)

    # Normalize weights to define a distribution
    weights /= np.sum(weights)

    # Sample points based on the computed weights
    indices = np.random.choice(np.arange(len(data)), size=Sample_size, replace=False, p=weights)
    coreset = data[indices]

    return coreset, weights  # Return coreset points and its weights.


coreset, weight = Sampling(data, k, centers, Sample_size)  # Call coreset construction algorithm (Sampling())

#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

#----Practical Coresets performance----# 	
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)						# Run weighted kMeans++ on coreset points
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)	# Run weighted KMeans on the coreset, using the inital centers from the above line.
Coreset_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution

#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	

print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset)