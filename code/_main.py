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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


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

##### Vera - Find missing values
df.info()
print(df.info())
# -> missing values only in weight_kg column; we do not use that, so no need to deal with the missing values

##### Oliver - Aggregate dataframe for country, year, category and flow

aggregated = (
    df.groupby(['country_or_area', 'year', 'category', 'flow'])['trade_usd']
      .sum()
      .reset_index(name='trade_usd_sum'))

# Pivot flows into columns
pivot = pd.pivot_table(
    aggregated,
    values='trade_usd_sum',
    index=['country_or_area', 'year', 'category'],
    columns=['flow'],
    aggfunc='sum',
    fill_value=0)

df_net = pivot.reset_index()

###### Oliver - Compute net trade and re_export and re_import ratio
df_net["net_usd"] = (df_net["Export"] - df_net["Import"]) + (df_net["Re-Export"] - df_net["Re-Import"])
df_net["net_imports"] = (df_net["Import"] + df_net["Re-Import"])
df_net["net_exports"] = (df_net["Export"] + df_net["Re-Export"])

df_net["reexport_ratio"] = df_net["Re-Export"] / df_net["net_exports"]
df_net["reimport_ratio"] = df_net["Re-Import"] / df_net["net_imports"]
df_net.loc[df_net["net_exports"] == 0, "reexport_ratio"] = 0
df_net.loc[df_net["net_imports"] == 0, "reimport_ratio"] = 0

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
#Vera: differentiate between categories, adjusting features and k

# K-Means for Cereals:

# 1. Aggregate per country
df_cluster_cereals = (
    df_net_cereals
    .groupby('country_or_area')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean()
    .reset_index())

# 2. Scale
scaler = StandardScaler()
cereals_scaled = scaler.fit_transform(
    df_cluster_cereals[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']])
print("cereals_scaled type:", type(cereals_scaled))
print("cereals_scaled shape:", cereals_scaled.shape)


# 3. Elbow method
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

# 4. Final K-Means (k = 7)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_cluster_cereals['cluster'] = kmeans.fit_predict(cereals_scaled)

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
    df_cluster_cereals
    .groupby('cluster')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(cluster_summary)

# 8. 2D Visualization
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

# 9. Merge clusters back to df_net
df_net = df_net.merge(
    df_cluster_cereals[['country_or_area', 'cluster']],
    on='country_or_area',
    how='left')

#Time-Series K-Means for cereals:

# Pivot to get time series: rows = countries, columns = years
ts_data = df_net_cereals.pivot(index='country_or_area', columns='year', values='net_usd').fillna(0)

print(ts_data.head())

scaler = StandardScaler()
ts_scaled = scaler.fit_transform(ts_data)

ts_scaled_3d = to_time_series_dataset(ts_scaled)
print(ts_scaled_3d.shape)  # (n_countries, n_years, 1)

# Choose number of clusters
n_clusters = 7

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

plt.title(f'Time-Series Clustering of Countries - {'10_cereals'}')
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
    how='left')

print(df_net[['country_or_area', 'cluster_ts']].drop_duplicates())


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
ironsteel_scaled = scaler.fit_transform(
    df_cluster[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']])
print("Iron&Steel_scaled type:", type(ironsteel_scaled))
print("Iron&Steel_scaled shape:", ironsteel_scaled.shape)


# 3. Elbow method
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

# 4. Final K-Means (k = 7)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(ironsteel_scaled)

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


# Time Series K Means for Iron&Steel

# Pivot to time-series format
ts_data = df_net_IronAndSteel.pivot(index='country_or_area', columns='year', values='net_usd').fillna(0)

# Scale each country's series
scaler = StandardScaler()
ts_scaled = scaler.fit_transform(ts_data)

# Convert to tslearn 3D format
ts_scaled_3d = to_time_series_dataset(ts_scaled)

# Time-Series K-Means clustering
n_clusters = 7
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

plt.title(f'Time-Series Clustering of Countries - {'72_iron_and_steel'}')
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
    how='left')

print(df_net[['country_or_area','cluster_ts_iron_steel']].drop_duplicates())

#List countries by clusters
for c in range(n_clusters):
    cluster_members = ts_data[ts_data['cluster_ts'] == c].drop(columns='cluster_ts')
    for country in cluster_members.index:
        plt.plot(cluster_members.columns, cluster_members.loc[country], alpha=0.5)

plt.title(f'Time-Series Clustering of Countries - {'72_iron_and_steel'}')
plt.xlabel('Year')
plt.ylabel('Net Trade (scaled)')
plt.show()

cluster_summary = ts_data.groupby('cluster_ts').apply(lambda x: list(x.index))
print(cluster_summary)


#Hierarchical Clustering: Agglomerative

# Agglomerative Clustering for Iron&Steel:

# Linkage matrix
Z = linkage(ironsteel_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=df_cluster['country_or_area'].values,
    leaf_rotation=90,
    leaf_font_size=8)

plt.title('Hierarchical Clustering Dendrogram (Ward) for Iron&Steel')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward')

df_cluster['cluster_agglomerative_ironsteel'] = agg.fit_predict(ironsteel_scaled)

print(df_cluster.head())

agglomerative_summary = (
    df_cluster
    .groupby('cluster_agglomerative_ironsteel')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean())

print(agglomerative_summary)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster,
    x='Import',
    y='Export',
    hue='cluster_agglomerative_ironsteel',
    palette='Set1')

plt.title('Hierarchical Clustering for Iron&Steel')
plt.xlabel('Import')
plt.ylabel('Export')
plt.grid(True)
plt.show()

# add the clusters as a column to the iron&steel dataset:
df_net_IronAndSteel = df_net_IronAndSteel.merge(
    df_cluster[['country_or_area', 'cluster_agglomerative_ironsteel']],
    on='country_or_area',
    how='left')

save_csv(df_cluster, "df_cluster.csv")
save_csv(df_net, "df_net_new.csv")
save_csv(df_net_IronAndSteel, "df_net_IronAndSteel.csv")

# list of clusters:
clusters_IronAndSteel = (df_cluster.groupby('cluster_agglomerative_ironsteel')['country_or_area']
    .apply(list))
for cluster, countries in clusters_IronAndSteel.items():
    print(f"Cluster {cluster}:")
    for c in countries:
        print(f"  - {c}")
    print()

# Time Series for Iron&Steel

# sort Data by year:
df_net_IronAndSteel = df_net_IronAndSteel.sort_values("year")

# Aggregate: one time series per cluster
# Because each cluster contains many countries, aggregate within each cluster at each time step
# most common: aggregate with the mean
cluster_ts_ironsteel = (
    df_net_IronAndSteel
    .groupby(["cluster_agglomerative_ironsteel", "year"])[
        [   "Export",
            "Import",
            "Re-Export",
            "Re-Import",
            "net_usd",
            "net_imports",
            "net_exports",
            "reexport_ratio",
            "reimport_ratio",]]
    .mean()
    .reset_index())

save_csv(cluster_ts_ironsteel, "cluster_ts_ironsteel.csv")
# this dataset gives us the time series data for each cluster in iron&steel

# plot time series for each cluster

clusters = sorted(
    cluster_ts_ironsteel["cluster_agglomerative_ironsteel"].unique())

level_features = ["Export", "Import", "Re-Export", "Re-Import", "net_usd"]
ratio_features = ["reexport_ratio", "reimport_ratio"]

for c1 in clusters:
    data = cluster_ts_ironsteel[
        cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == c1]

    plt.figure(figsize=(10, 5))

    for feature in level_features:
        plt.plot(data["year"], data[feature], label=feature)

    plt.title(f"Iron & Steel - Level – Cluster {c1}")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

for c1 in clusters:
    data = cluster_ts_ironsteel[
        cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == c1]

    plt.figure(figsize=(10, 5))

    for feature in ratio_features:
        plt.plot(data["year"], data[feature], label=feature)

    plt.title(f"Iron & Steel - Ratio – Cluster {c1}")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Agglomerative Clustering for Cereals:

# Linkage matrix
X = linkage(cereals_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    X,
    labels=df_cluster_cereals['country_or_area'].values,
    leaf_rotation=90,
    leaf_font_size=8)

plt.title('Hierarchical Clustering Dendrogram (Ward) for Cereals')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')

df_cluster_cereals['cluster_agglomerative_cereals'] = agg.fit_predict(cereals_scaled)

print(df_cluster_cereals)

# list of clusters:
clusters_cereals = (
    df_cluster_cereals
    .groupby('cluster_agglomerative_cereals')['country_or_area']
    .apply(list))

for cluster, countries in clusters_cereals.items():
    print(f"Cluster {cluster}:")
    for c in countries:
        print(f"  - {c}")
    print()

agglomerative_summary = (
    df_cluster_cereals
    .groupby('cluster_agglomerative_cereals')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean()).apply(list)

print(agglomerative_summary)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster_cereals,
    x='Import',
    y='Export',
    hue='cluster_agglomerative_cereals',
    palette='Set1')

plt.title('Hierarchical Clustering for Cereals')
plt.xlabel('Import')
plt.ylabel('Export')
plt.grid(True)
plt.show()

# add the clusters as a column to the cereals dataset:
df_net_cereals = df_net_cereals.merge(
    df_cluster_cereals[['country_or_area', 'cluster_agglomerative_cereals']],
    on='country_or_area',
    how='left')
save_csv(df_net_cereals, "df_net_cereals.csv")


# Time Series for Cereals

# sort Data by year:
df_net_cereals = df_net_cereals.sort_values("year")

# Aggregate: one time series per cluster
cluster_ts_cereals = (
    df_net_cereals
    .groupby(["cluster_agglomerative_cereals", "year"])[
        [   "Export",
            "Import",
            "Re-Export",
            "Re-Import",
            "net_usd",
            "net_imports",
            "net_exports",
            "reexport_ratio",
            "reimport_ratio",]]
    .mean()
    .reset_index())

save_csv(cluster_ts_cereals, "cluster_ts_cereals.csv")
# this dataset gives us the time series data for each cluster in cereals

# plot time series for each cluster

clusters_cereals = sorted(
    cluster_ts_cereals["cluster_agglomerative_cereals"].unique())

for c1 in clusters:
    data = cluster_ts_cereals[
        cluster_ts_cereals["cluster_agglomerative_cereals"] == c1]

    plt.figure(figsize=(10, 5))

    for feature in level_features:
        plt.plot(data["year"], data[feature], label=feature)

    plt.title(f"Cereals - Level – Cluster {c1}")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

for c1 in clusters:
    data = cluster_ts_cereals[
        cluster_ts_cereals["cluster_agglomerative_cereals"] == c1]

    plt.figure(figsize=(10, 5))

    for feature in ratio_features:
        plt.plot(data["year"], data[feature], label=feature)

    plt.title(f"Cereals - Ratio – Cluster {c1}")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# FORECASTING:

#Cluster 0 iron and steel

# --- 1. PREPARE THE NORMALIZED TIME SERIES ---
cluster_0_raw = df_net_IronAndSteel[df_net_IronAndSteel["cluster_agglomerative_ironsteel"] == 0].copy()

# Calculate Normalized Trade Balance for each country-year
# We add a small epsilon (1e-6) to avoid division by zero if trade is ever exactly 0
cluster_0_raw['ntb_ratio'] = (cluster_0_raw['net_usd']) / (cluster_0_raw['Import'] + cluster_0_raw['Export'] + 1e-6)

# Now aggregate the RATIO by year (Mean of ratios)
ts_ratio = cluster_0_raw.groupby("year")["ntb_ratio"].mean().sort_index()
ts_ratio.index = ts_ratio.index.astype(int)

# --- 2. FORECAST HORIZON ---
h = 12

# --- 3. FIT HOLT'S LINEAR MODEL ---
# Using the ratio series (ts_ratio)
model = ExponentialSmoothing(
    ts_ratio,
    trend="add",
    seasonal=None).fit()

# --- 4. GENERATE FORECAST ---
forecast_values = model.forecast(h)
last_year = ts_ratio.index.max()
future_years = range(last_year + 1, last_year + h + 1)

forecast = pd.Series(
    [ts_ratio.iloc[-1]] + list(forecast_values.values),
    index=[last_year] + list(future_years))

# --- 5. PLOT THE RESULTS ---
plt.figure(figsize=(10, 5))
plt.plot(ts_ratio.index, ts_ratio, label="Observed (Avg NTB Ratio)", color='teal', linewidth=2)
plt.plot(forecast.index, forecast, label="Forecasted Ratio", color='orange', linestyle="--", linewidth=2)

plt.axhline(0, color='black', linewidth=1, alpha=0.5) # The "Balance" line
plt.title("Iron & Steel – Cluster 0 – Structural Trade Balance Forecast")
plt.xlabel("Year")
plt.ylabel("NTB Ratio (Exports-Imports / Total Trade)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# 1. Filter data for Cluster 0 and set Year as index
c0_data = cluster_ts_ironsteel[
    cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 0
    ].sort_values("year").set_index("year")

# Define our two groups based on your methodology
level_features = ["Export", "Import", "net_usd"]
ratio_features = ["reexport_ratio", "reimport_ratio"]
h = 12  # Forecast horizon (up to 2028ish)

# --- A. FORECASTING LEVELS (Holt's Linear Trend) ---
plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c0_data[feature]

    # Levels represent scale/volume; we use an additive trend
    model = ExponentialSmoothing(ts, trend="add", seasonal=None).fit()
    forecast_values = model.forecast(h)

    # Plotting
    line, = plt.plot(ts.index, ts, label=f"Observed {feature}")
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Iron & Steel – Cluster 0 – Level Features Forecast (USD)")
plt.ylabel("Value (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- B. FORECASTING RATIOS (Simple Exponential Smoothing) ---
plt.figure(figsize=(12, 6))
for feature in ratio_features:
    ts = c0_data[feature]

# Ratios represent structure/identity; SES is better for mean-reversion
    model = SimpleExpSmoothing(ts).fit()
    forecast_values = model.forecast(h)

# Plotting
    line, = plt.plot(ts.index, ts, label=f"Observed {feature}")
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Iron & Steel – Cluster 0 – Structural Ratio Forecast")
plt.ylabel("Ratio (0 to 1)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Forecast for cluster 1 for iron and steel

# 1. Filter specifically for Cluster 1
c1_data = cluster_ts_ironsteel[
    cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 1
    ].sort_values("year").set_index("year")

# Re-using your defined groups
level_features = ["Export", "Import", "net_usd"]
ratio_features = ["reexport_ratio", "reimport_ratio"]
h = 12

# --- A. FORECASTING LEVELS (Holt's Linear Trend) ---
# For producers, we use damped_trend=True to keep growth projections realistic
plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c1_data[feature]

    # Use additive trend with damping for these high-volume exporters
    model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
    forecast_values = model.forecast(h)

    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Iron & Steel – Cluster 1 – Producer Level Forecast (Scale: 1e9 USD)")
plt.ylabel("Value (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- B. FORECASTING RATIOS (Simple Exponential Smoothing) ---
plt.figure(figsize=(12, 6))
for feature in ratio_features:
    ts = c1_data[feature]

    # SES is best here because Cluster 1 ratios are near-zero and noisy
    model = SimpleExpSmoothing(ts).fit()
    forecast_values = model.forecast(h)

    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Iron & Steel – Cluster 1 – Producer Ratio Forecast")
plt.ylabel("Ratio Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Forecast for Cluster 2
# 1. Filter specifically for Cluster 2
c2_data = cluster_ts_ironsteel[
    cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 2
    ].sort_values("year").set_index("year")

# Re-using your defined groups to maintain consistency
level_features = ["Export", "Import", "net_usd"]
ratio_features = ["reexport_ratio", "reimport_ratio"]
h = 12

# --- A. FORECASTING LEVELS (Damped Holt's) ---
# Damping is essential here to handle the extreme 2011-2013 volatility
plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c2_data[feature]

    # We use damped_trend=True because the spikes in Cluster 2 are likely outliers
    model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
    forecast_values = model.forecast(h)

    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Iron & Steel – Cluster 2 – Volatile Level Forecast (USD)")
plt.ylabel("Value (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Forecast for Cluster 0 – Cereals

ts = (cluster_ts_cereals[cluster_ts_cereals["cluster_agglomerative_cereals"] == 0]
    .sort_values("year")
    .set_index("year")["net_usd"])

ts.index = ts.index.astype(int)

h = 15

model = ExponentialSmoothing(
    ts,
    trend="add",
    seasonal=None).fit()

forecast_values = model.forecast(h)

last_year = ts.index.max()
future_years = range(last_year + 1, last_year + h + 1)

forecast = pd.Series(
    forecast_values.values,
    index=future_years)

plt.figure(figsize=(9,5))
plt.plot(ts.index, ts, label="Observed", linewidth=2)
plt.plot(forecast.index, forecast, linestyle="--", label="Forecast", linewidth=2)

plt.axvline(x=last_year, color="gray", linestyle=":", alpha=0.6)

plt.title("Cereals – Cluster 0 – net_usd Forecast")
plt.xlabel("Year")
plt.ylabel("USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


