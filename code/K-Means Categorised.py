from pandas import read_csv, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn import preprocessing
from pathlib import Path


##### Oliver Save CSV function
def save_csv(dataframe: DataFrame, name_of_csv: str):
    dataframe.to_csv(CSV_DIR / name_of_csv, index=False)


##### Oliver - Read CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)

df = read_csv(CSV_DIR / "case2.csv", sep=";")

print("Initial Dataset:")
print(df.info())


##### Paula - Clean & Remove Columns (Remove EU28, Convert Federal Rep. of Germany to Germany)
df = df.drop(columns=["commodity", "comm_code"])
df = df[~df["country_or_area"].isin([
    "EU-28",
    "So. African Customs Union"])]
df["country_or_area"] = df["country_or_area"].replace(
    "Fmr Fed. Rep. of Germany", "Germany")

print("\nDataset after dropping columns \"commodity\" and \"comm_code\" and removing EU28, Converting Federal Rep. of Germany to Germany:")
print(df.info())


##### Oliver - Check if stratified sample already exists
sample_path = CSV_DIR / "case2_sampled.csv"

if sample_path.exists():
    df = read_csv(sample_path, sep=",")
    df = df.reset_index(drop=True)
else:
##### Vera - Stratified Sampling of Countries as Classes
    df = df.groupby('country_or_area', group_keys=False).apply(
        lambda x: x.sample(frac=0.4))
    df = df.reset_index(drop=True)
    save_csv(df, "case2_sampled.csv")

print("\nDataset after stratified sampling of countries as classes:")
print(df.info())

##### Vera - Deal with missing values in weight category
df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].median())

print("\nDataset after dealing with missing values in weight category:")
print(df.info())

##### Vera - Create Dummy Variables for Category & Flow
df_dummy = (pd.get_dummies(df, columns = ['category', 'flow'], prefix_sep='_', dummy_na=False, dtype='int'))
save_csv(df_dummy, "dummy.csv")
print("\nDataset after creating dummy variables for category & flow:")
print(df_dummy.info())


##### Oliver - Create net (import, export, reimport, reexport) values (4) for each year, country and category

aggregated = (
    df.groupby(['country_or_area', 'year', 'category', 'flow'], as_index=False)
      .agg(trade_usd_sum=('trade_usd', 'sum'))
)

# Pivot flows into columns
pivot = aggregated.pivot_table(
    index=['country_or_area', 'year', 'category'],
    columns='flow',
    values='trade_usd_sum',
    fill_value=0
)

df_net = pivot.reset_index()

###### Oliver - Compute net trade and re_export and re_import ratio
df_net["net_usd"] = (df_net["Export"] - df_net["Import"]) + (df_net["Re-Export"] - df_net["Re-Import"])
df_net["net_imports"] = (df_net["Import"] + df_net["Re-Import"])
df_net["net_exports"] = (df_net["Export"] + df_net["Re-Export"])

df_net["reexport_ratio"] = df_net["Re-Export"] / df_net["net_exports"]
df_net["reimport_ratio"] = df_net["Re-Import"] / df_net["net_imports"]
df_net.loc[df_net["net_exports"] == 0, "reexport_ratio"] = 0
df_net.loc[df_net["net_imports"] == 0, "reimport_ratio"] = 0

#Vera: compute export/ import ratio
df_net["export_import_ratio"] = df_net["net_exports"] / df_net["net_imports"]

# Save
save_csv(df_net, "net_trade_by_flow.csv")

print("\nNet trade dataset created:")
print(df_net.info())
print(df_net.describe())


# Vera: filter df_net for categories cereals and iron&steel:
df_net_cereals = df_net.loc[df_net['category'] == '10_cereals', :]
print(df_net_cereals.head())

df_net_IronAndSteel = df_net.loc[df_net['category'] == '72_iron_and_steel', :]
print(df_net_IronAndSteel.head())


#Thao: K-Means
# =======
#Vera: K-Means for Cereals

# 1. Aggregate per country
df_cluster = (
    df_net_cereals
    .groupby('country_or_area')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean()
    .reset_index())

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    df_cluster[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']])
print("X_scaled type:", type(X_scaled))
print("X_scaled shape:", X_scaled.shape)


# 3. Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Optimal k for Cereals')
plt.grid(True)
plt.show()

# 4. Final K-Means (k = 7)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Cluster centroids in ORIGINAL units
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio'])

print("Cluster centroids (original scale):")
print(centroids)

# 6. Euclidean distances between cluster centers (scaled space)
Euclidean = pd.DataFrame(
    pairwise_distances(kmeans.cluster_centers_, metric='euclidean'))

print("Euclidean distances between clusters:")
print(Euclidean)

# 7. Cluster summary (same as centroids but easier to explain)
cluster_summary = (
    df_cluster
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(cluster_summary)

# 8. 2D Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster,
    x='Import',
    y='Export',
    hue='cluster',
    palette='Set2')
plt.title('K-Means Clustering of Countries by Trade for Cereals')
plt.xlabel('Imports')
plt.ylabel('Exports')
plt.grid(True)
plt.show()

# 9. Merge clusters back to df_net
df_net = df_net.merge(
    df_cluster[['country_or_area', 'cluster']],
    on='country_or_area',
    how='left')

#K-Means HeatMap
# Cluster-level averages
heatmap_data = (
    df_cluster
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(heatmap_data)

scaler_hm = StandardScaler()
heatmap_scaled = pd.DataFrame(
    scaler_hm.fit_transform(heatmap_data),
    index=heatmap_data.index,
    columns=heatmap_data.columns)

print(heatmap_scaled)

plt.figure(figsize=(8, 5))
sns.heatmap(
    heatmap_scaled,
    annot=True,
    cmap='coolwarm',
    center=0,
    linewidths=0.5)

plt.title('Heatmap: K-Means, Cereals)')
plt.xlabel('Trade Indicators')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

sns.clustermap(
    heatmap_scaled,
    cmap='coolwarm',
    center=0,
    annot=True,
    figsize=(8, 6))


#K-Means for Iron&Steel

# 1. Aggregate per country
df_cluster = (
    df_net_IronAndSteel
    .groupby('country_or_area')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean()
    .reset_index())

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    df_cluster[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']])
print("X_scaled type:", type(X_scaled))
print("X_scaled shape:", X_scaled.shape)


# 3. Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Optimal k for Iron&Steel')
plt.grid(True)
plt.show()

# 4. Final K-Means (k = 7)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Cluster centroids in ORIGINAL units
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio'])

print("Cluster centroids (original scale):")
print(centroids)

# 6. Euclidean distances between cluster centers (scaled space)
Euclidean = pd.DataFrame(
    pairwise_distances(kmeans.cluster_centers_, metric='euclidean'))

print("Euclidean distances between clusters:")
print(Euclidean)

# 7. Cluster summary (same as centroids but easier to explain)
cluster_summary = (
    df_cluster
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(cluster_summary)

# 8. 2D Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster,
    x='Import',
    y='Export',
    hue='cluster',
    palette='Set2')
plt.title('K-Means Clustering of Countries by Trade for Iron&Steel')
plt.xlabel('Imports')
plt.ylabel('Exports')
plt.grid(True)
plt.show()

# 9. Merge clusters back to df_net
df_net = df_net.merge(
    df_cluster[['country_or_area', 'cluster']],
    on='country_or_area',
    how='left')

#K-Means HeatMap
# Cluster-level averages
heatmap_data = (
    df_cluster
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(heatmap_data)

scaler_hm = StandardScaler()
heatmap_scaled = pd.DataFrame(
    scaler_hm.fit_transform(heatmap_data),
    index=heatmap_data.index,
    columns=heatmap_data.columns)

print(heatmap_scaled)

plt.figure(figsize=(8, 5))
sns.heatmap(
    heatmap_scaled,
    annot=True,
    cmap='coolwarm',
    center=0,
    linewidths=0.5)

plt.title('Heatmap: K-Means, Cereals)')
plt.xlabel('Trade Indicators')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

sns.clustermap(
    heatmap_scaled,
    cmap='coolwarm',
    center=0,
    annot=True,
    figsize=(8, 6))