from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances
import seaborn as sns

import dataprep
dataprep.dataprep()
df_net = dataprep.df_net
df_net_ironsteel = dataprep.df_net_ironsteel
df_net_cereals = dataprep.df_net_cereals
ironsteel_scaled = dataprep.ironsteel_scaled
cereals_scaled = dataprep.cereals_scaled
scaler_ironsteel = dataprep.scaler_ironsteel
scaler_cereals = dataprep.scaler_cereals
df_cluster_ironsteel = dataprep.df_cluster_ironsteel
df_cluster_cereals = dataprep.df_cluster_cereals

##### Thao & Vera: K-Means

# K-Means for Cereals:

# Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cereals_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Optimal k for Cereals')
plt.grid(True)
plt.show()

# Final K-Means (k = 7)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_cluster_cereals['cluster'] = kmeans.fit_predict(cereals_scaled)

# Cluster centroids in ORIGINAL units
centroids = pd.DataFrame(
    scaler_cereals.inverse_transform(kmeans.cluster_centers_),
    columns=['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio'])

print("Cluster centroids (original scale):")
print(centroids)

# Euclidean distances between cluster centers (scaled space)
Euclidean = pd.DataFrame(
    pairwise_distances(kmeans.cluster_centers_, metric='euclidean'))

print("Euclidean distances between clusters:")
print(Euclidean)

# Cluster summary (same as centroids but easier to explain)
cluster_summary = (
    df_cluster_cereals
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(cluster_summary)

# 2D Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster_cereals,
    x='Import',
    y='Export',
    hue='cluster',
    palette='Set2')
plt.title('K-Means Clustering of Countries by Trade for Cereals')
plt.xlabel('Imports')
plt.ylabel('Exports')
plt.grid(True)
plt.show()


#K-Means for Iron&Steel

# Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(ironsteel_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Optimal k for Iron&Steel')
plt.grid(True)
plt.show()

# Final K-Means (k = 7)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_cluster_ironsteel['cluster'] = kmeans.fit_predict(ironsteel_scaled)

# Cluster centroids in ORIGINAL units
centroids = pd.DataFrame(
    scaler_ironsteel.inverse_transform(kmeans.cluster_centers_),
    columns=['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio'])

print("Cluster centroids (original scale):")
print(centroids)

# Euclidean distances between cluster centers (scaled space)
Euclidean = pd.DataFrame(
    pairwise_distances(kmeans.cluster_centers_, metric='euclidean'))

print("Euclidean distances between clusters:")
print(Euclidean)

# Cluster summary (same as centroids but easier to explain)
cluster_summary = (
    df_cluster_ironsteel
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(cluster_summary)

# 2D Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster_ironsteel,
    x='Import',
    y='Export',
    hue='cluster',
    palette='Set2')
plt.title('K-Means Clustering of Countries by Trade for Iron&Steel')
plt.xlabel('Imports')
plt.ylabel('Exports')
plt.grid(True)
plt.show()