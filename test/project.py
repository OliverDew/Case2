from pandas import read_csv
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


##### Oliver - Read CSV
df = read_csv("case2.csv", sep=";")

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


##### Vera - Stratified Sampling of Countries as Classes
df = df.groupby('country_or_area', group_keys=False).apply(
    lambda x: x.sample(frac=0.4))
df = df.reset_index(drop=True)
df.to_csv("case2_sampled.csv", index=False)

print("\nDataset after stratified sampling of countries as classes:")
print(df.info())

##### Vera - Deal with missing values in weight category
df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].median())

print("\nDataset after dealing with missing values in weight category:")
print(df.info())

##### Vera - Create Dummy Variables for Category & Flow
df_dummy = (pd.get_dummies(df, columns = ['category', 'flow'], prefix_sep='_', dummy_na=False, dtype='int'))
df_dummy.to_csv("dummy.csv", index=False)
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

# Compute net trade
df_net["net_usd"] = (df_net["Export"] - df_net["Import"]) + (df_net["Re-Export"] - df_net["Re-Import"])
df_net["net_imports"] = (df_net["Import"] + df_net["Re-Import"])
df_net["net_export"] = (df_net["Export"] + df_net["Re-Export"])

# Save
df_net.to_csv("case2_net_trade_by_flow.csv", index=False)

print("\nNet trade dataset created:")
print(df_net.info())
print(df_net.describe())

# Oliver - Plot Albanian cereal production over the years
alb_cereal = df_net[
    (df_net["country_or_area"] == "Albania") &
    (df_net["category"] == "10_cereals")
]

fig, ax = plt.subplots(figsize=(12, 8))
plt.bar(alb_cereal["year"], alb_cereal["net_usd"], edgecolor="black", alpha=0.5, label="Albania")

ax.ticklabel_format(style='plain', axis='y')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.title("Albania – Net Trade in Cereals Over Time")
plt.xlabel("Year")
plt.ylabel("Net Trade (USD)")
plt.grid(True)
plt.show()

# Vera: filter df_net for categories cereals and iron&steel:
df_net_cereals = df_net.loc[df_net['category'] == '10_cereals', :]
print(df_net_cereals.head())

df_net_IronAndSteel = df_net.loc[df_net['category'] == '72_iron_and_steel', :]
print(df_net_IronAndSteel.head())

# Vera: scatter plot net_usd per year:
fig, ax = plt.subplots(figsize=(12, 8))
df_net.plot.scatter(x="year", y="net_usd", ax=ax)
plt.title("Net_USD per Year")

# Vera: Bar Chart Top Players in Cereals
fig, ax = plt.subplots(figsize=(10, 6))
df_net_cereals['country_or_area'].value_counts().plot.bar(color='skyblue', edgecolor='black', ax=ax)
plt.title('Trade in Cereals')
plt.xlabel('country_or_area')
plt.ylabel('net_usd')

# Vera: Bar Chart Top Players in Iron&Steel
fig, ax = plt.subplots(figsize=(10, 6))
df_net_IronAndSteel['country_or_area'].value_counts().plot.bar(color='skyblue', edgecolor='black', ax=ax)
plt.title('Trade in Iron and Steel')
plt.xlabel('country_or_area')
plt.ylabel('net_usd')
plt.xlabel('country_or_area')
plt.ylabel('net_usd')

# Vera: Pie Chart for Category
fig, ax = plt.subplots(figsize=(8, 8))
df_net['category'].value_counts().plot.pie(autopct='%1.0f%%')
plt.title('Proportion of Cereals and Iron and Steel')
plt.ylabel('')

# Vera: Boxplot
fig, ax = plt.subplots(figsize=(12, 8))
df_net.boxplot(column='net_usd', by='category', ax=ax)
plt.xlabel('category')
plt.ylabel('net_usd')

# Vera: Scatter Plot by Category
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='net_imports', y='net_export', hue='country_or_area', data=df_net)
plt.title('Imports vs. Exports Colored by Country')
plt.xlabel('net_imports')
plt.ylabel('net_export')
plt.legend(title='Country')

plt.show()

#Thao: K-Means
# =======
#K-Means

# 1. Aggregate per country
df_cluster = (
    df_net
    .groupby('country_or_area')[['net_imports', 'net_export', 'net_usd']]
    .mean()
    .reset_index())

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    df_cluster[['net_imports', 'net_export', 'net_usd']]
)

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
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# 4. Final K-Means (k = 3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Cluster centroids in ORIGINAL units
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=['net_imports', 'net_export', 'net_usd']
)

print("Cluster centroids (original scale):")
print(centroids)

# 6. Euclidean distances between cluster centers (scaled space)
Euclidean = pd.DataFrame(
    pairwise_distances(kmeans.cluster_centers_, metric='euclidean')
)

print("Euclidean distances between clusters:")
print(Euclidean)

# 7. Cluster summary (same as centroids but easier to explain)
cluster_summary = (
    df_cluster
    .groupby('cluster')[['net_imports', 'net_export', 'net_usd']]
    .mean()
)

print(cluster_summary)

# 8. 2D Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster,
    x='net_imports',
    y='net_export',
    hue='cluster',
    palette='Set2'
)
plt.title('K-Means Clustering of Countries by Trade')
plt.xlabel('Average Net Imports')
plt.ylabel('Average Net Exports')
plt.grid(True)
plt.show()

# 9. Merge clusters back to df_net
df_net = df_net.merge(
    df_cluster[['country_or_area', 'cluster']],
    on='country_or_area',
    how='left'
)
#K-Means HeatMap
# Cluster-level averages
heatmap_data = (
    df_cluster
    .groupby('cluster')[['net_imports', 'net_export', 'net_usd']]
    .mean()
)

print(heatmap_data)


scaler_hm = StandardScaler()
heatmap_scaled = pd.DataFrame(
    scaler_hm.fit_transform(heatmap_data),
    index=heatmap_data.index,
    columns=heatmap_data.columns
)

print(heatmap_scaled)

plt.figure(figsize=(8, 5))
sns.heatmap(
    heatmap_scaled,
    annot=True,
    cmap='coolwarm',
    center=0,
    linewidths=0.5
)

plt.title('Cluster Heatmap: Trade Indicators (K-Means)')
plt.xlabel('Trade Indicators')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()

sns.clustermap(
    heatmap_scaled,
    cmap='coolwarm',
    center=0,
    annot=True,
    figsize=(8, 6)
)

#Hierarchical Clustering: Agglomerative

# Linkage matrix
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=df_cluster['country_or_area'].values,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

df_cluster['cluster_hier'] = agg.fit_predict(X_scaled)

print(df_cluster.head())

hier_summary = (
    df_cluster
    .groupby('cluster_hier')[['net_imports', 'net_export', 'net_usd']]
    .mean()
)

print(hier_summary)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster,
    x='net_imports',
    y='net_export',
    hue='cluster_hier',
    palette='Set1'
)

plt.title('Hierarchical Clustering of Countries by Trade')
plt.xlabel('Average Net Imports')
plt.ylabel('Average Net Exports')
plt.grid(True)
plt.show()

df_net = df_net.merge(
    df_cluster[['country_or_area', 'cluster_hier']],
    on='country_or_area',
    how='left'
)

#Time-Series K-Means for cereals
import pandas as pd
import numpy as np

# Select category (e.g., cereals)
category = '10_cereals'
df_cat = df_net[df_net['category'] == category]

# Pivot to get time series: rows = countries, columns = years
ts_data = df_cat.pivot(index='country_or_area', columns='year', values='net_usd').fillna(0)

print(ts_data.head())


scaler = StandardScaler()
ts_scaled = scaler.fit_transform(ts_data)


ts_scaled_3d = to_time_series_dataset(ts_scaled)
print(ts_scaled_3d.shape)  # (n_countries, n_years, 1)

# Choose number of clusters
n_clusters = 3

km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=50, random_state=42)
labels = km.fit_predict(ts_scaled_3d)

# Add cluster labels back to dataframe
ts_data['cluster_ts'] = labels
print(ts_data[['cluster_ts']])

plt.figure(figsize=(12,6))

for c in range(n_clusters):
    cluster_members = ts_data[ts_data['cluster_ts'] == c].drop(columns='cluster_ts')
    for country in cluster_members.index:
        plt.plot(cluster_members.columns, cluster_members.loc[country], alpha=0.5)
    plt.plot(cluster_members.columns, cluster_members.mean(), linewidth=3, label=f'Cluster {c} Mean')

plt.title(f'Time-Series Clustering of Countries - {category}')
plt.xlabel('Year')
plt.ylabel('Net Trade (scaled)')
plt.legend()
plt.show()

for c in range(n_clusters):
    cluster_countries = ts_data[ts_data['cluster_ts'] == c].index.tolist()
    print(f"\nCluster {c}:")
    print(cluster_countries)

df_net = df_net.merge(
    ts_data['cluster_ts'].reset_index(),
    left_on='country_or_area',
    right_on='country_or_area',
    how='left'
)

print(df_net[['country_or_area', 'cluster_ts']].drop_duplicates())

plt.figure(figsize=(12,6))

# Select iron & steel category
category = '72_iron_and_steel'
df_cat = df_net[df_net['category'] == category]

# Pivot to time-series format
ts_data = df_cat.pivot(index='country_or_area', columns='year', values='net_usd').fillna(0)

# Scale each country's series
scaler = StandardScaler()
ts_scaled = scaler.fit_transform(ts_data)

# Convert to tslearn 3D format
ts_scaled_3d = to_time_series_dataset(ts_scaled)

# Time-Series K-Means clustering
n_clusters = 3
km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=50, random_state=42)
labels = km.fit_predict(ts_scaled_3d)

# Add cluster labels back to pivoted dataframe
ts_data['cluster_ts'] = labels

# Plot clusters
plt.figure(figsize=(12,6))
for c in range(n_clusters):
    cluster_members = ts_data[ts_data['cluster_ts'] == c].drop(columns='cluster_ts')
    for country in cluster_members.index:
        plt.plot(cluster_members.columns, cluster_members.loc[country], alpha=0.5)
    plt.plot(cluster_members.columns, cluster_members.mean(), linewidth=3, label=f'Cluster {c} Mean')

plt.title(f'Time-Series Clustering of Countries - {category}')
plt.xlabel('Year')
plt.ylabel('Net Trade (scaled)')
plt.legend()
plt.show()

# Print countries per cluster
for c in range(n_clusters):
    cluster_countries = ts_data[ts_data['cluster_ts'] == c].index.tolist()
    print(f"Cluster {c}:")
    print(cluster_countries)

# Merge cluster labels back to main df_net
df_net = df_net.merge(
    ts_data['cluster_ts'].reset_index().rename(columns={'cluster_ts':'cluster_ts_iron_steel'}),
    left_on='country_or_area',
    right_on='country_or_area',
    how='left'
)

print(df_net[['country_or_area','cluster_ts_iron_steel']].drop_duplicates())

#List countries by clusters
for c in range(n_clusters):
    cluster_members = ts_data[ts_data['cluster_ts'] == c].drop(columns='cluster_ts')
    for country in cluster_members.index:
        plt.plot(cluster_members.columns, cluster_members.loc[country], alpha=0.5)
# Add country name at the end of the line
plt.text(cluster_members.columns[-1], cluster_members.loc[country][-1], country, fontsize=8)
plt.plot(cluster_members.columns, cluster_members.mean(), linewidth=3, label=f'Cluster {c} Mean')

plt.title(f'Time-Series Clustering of Countries - {category}')
plt.xlabel('Year')
plt.ylabel('Net Trade (scaled)')
plt.legend()
plt.show()

cluster_summary = ts_data.groupby('cluster_ts').apply(lambda x: list(x.index))
print(cluster_summary)
