import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
sns.set(style="white", color_codes=True)
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')
dataset = dataset.fillna(dataset.mean())

x = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]]
y = dataset.iloc[:,12]




##building the model
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,
                    max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("KMeans:",score)

#PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)

x_scaler = scaler.transform(x)
pca = PCA(16)
x_pca = pca.fit_transform(x_scaler)
PCA_components = pd.DataFrame(x_pca)


features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

top_n_components = 2
x = PCA_components.iloc[:,:top_n_components]

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,
                    max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

nclusters = 4 # this is the k in kmeans

km = KMeans(n_clusters=nclusters)
km.fit(x)

y_cluster_kmeans = km.predict(x)

score = metrics.silhouette_score(x, y_cluster_kmeans)
print("PCA:", score)

plt.scatter(x[0], x[1], c=y_cluster_kmeans, s=50, cmap='viridis')
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

# df2 = pd.DataFrame(data=x_pca)
# finaldf = pd.concat([df2,dataset[['CREDIT_LIMIT']]],axis=1)
# print(finaldf)