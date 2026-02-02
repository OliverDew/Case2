from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

##### Oliver - Read CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)

df = read_csv(CSV_DIR / "case2.csv", sep=";")

print("Initial Dataset:")
print(df.info())


##### Oliver - Helpers
# Save CSV function
def save_csv(dataframe: DataFrame, name_of_csv: str):
    dataframe.to_csv(CSV_DIR / name_of_csv, index=False)

# Heatmap helper
def plot_heatmap(
    df_cluster: pd.DataFrame,
    scaled: np.ndarray,
    cluster_col: str,
    feature_cols: list[str],
    title: str,
    sort_clusters: bool = True,
):
    """
    Heatmap with rows=clusters and cols=features.
    Values are z-score cluster means
    """

    # build scaled dataframe per country
    X = pd.DataFrame(
        scaled,
        columns=feature_cols
    )
    X[cluster_col] = df_cluster[cluster_col].to_numpy()

    # cluster means in scaled space
    profile = X.groupby(cluster_col)[feature_cols].mean()

    if sort_clusters:
        profile = profile.sort_index()

    plt.figure(figsize=(9, 3 + 0.6 * len(profile)))
    sns.heatmap(profile, cmap="mako_r", center=0, annot=True, fmt=".2f")
    plt.title(title)
    plt.xlabel("Features (scaled mean)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

# Helpers for testing the cluster stability
def _nearest_centroid_assign(X_B: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each row in X_B to the nearest centroid (Euclidean distance).
    Returns labels in [0..k-1] corresponding to centroid index.
    """
    # distances: shape (n_B, k)
    dists = np.linalg.norm(X_B[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1)


def _hungarian_cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cluster labels are arbitrary. This finds the best 1-1 mapping between y_pred and y_true
    using the Hungarian algorithm, then returns accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    # maximize matches => minimize negative matches
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = cm[row_ind, col_ind].sum()
    return matched / cm.sum()


def cluster_stability_one_split(
    X_scaled: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
    test_size: float = 0.5,
    random_state: int = 42,
):
    """
    Implements the screenshot:
    1) split into A/B
    2) cluster A (hierarchical)
    3) compute A centroids and assign B to nearest centroid
    4) cluster FULL data
    5) compare labels for B: assigned-from-A vs full-cluster labels

    Returns:
      - ARI on B (permutation-invariant, recommended)
      - best-mapped accuracy on B (interpretability)
    """
    n = X_scaled.shape[0]
    idx_all = np.arange(n)

    idx_A, idx_B = train_test_split(
        idx_all, test_size=test_size, random_state=random_state, shuffle=True
    )

    X_A = X_scaled[idx_A]
    X_B = X_scaled[idx_B]

    # Step 2: cluster A
    model_A = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels_A = model_A.fit_predict(X_A)

    # Step 3: centroids from A clusters (in scaled feature space)
    centroids = np.vstack([X_A[labels_A == k].mean(axis=0) for k in range(n_clusters)])

    # Assign B to nearest centroid
    labels_B_assigned = _nearest_centroid_assign(X_B, centroids)

    # Step 4: cluster full dataset
    model_full = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels_full = model_full.fit_predict(X_scaled)

    # Step 5: compare on B only
    labels_B_full = labels_full[idx_B]

    ari = adjusted_rand_score(labels_B_full, labels_B_assigned)
    acc = _hungarian_cluster_accuracy(labels_B_full, labels_B_assigned)

    return ari, acc


def cluster_stability_repeated(
    X_scaled: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
    test_size: float = 0.5,
    n_repeats: int = 30,
    seed: int = 42,
):
    """
    Repeats the split procedure to avoid one lucky/unlucky split.
    Returns a DataFrame with ARI + best-mapped accuracy per split and prints summary.
    """
    results = []
    for i in range(n_repeats):
        rs = seed + i
        ari, acc = cluster_stability_one_split(
            X_scaled=X_scaled,
            n_clusters=n_clusters,
            linkage=linkage,
            test_size=test_size,
            random_state=rs,
        )
        results.append({"split": i, "ARI_B": ari, "ACC_B_mapped": acc})

    res = pd.DataFrame(results)
    print("\nCluster stability (repeated splits)")
    print(res.describe()[["ARI_B", "ACC_B_mapped"]])

    return res

##### Oliver - Delete all CSVs except case2 + sample on each run
for file in CSV_DIR.glob("*.csv"):
    if file.name not in {"case2.csv", "case2_sampled.csv"}:
        file.unlink()


##### Paula - Clean and Remove Columns
df = df.drop(columns=["commodity", "comm_code"])
df = df[~df["country_or_area"].isin([
    "EU-28",
    "So. African Customs Union",
    "Other Asia, nes"])]
df["country_or_area"] = df["country_or_area"].replace(
    "Fmr Fed. Rep. of Germany", "Germany")
df["country_or_area"] = df["country_or_area"].replace(
    "Fmr Sudan", "Sudan")

print("\nDataset after dropping columns \"commodity\" and \"comm_code\" and removing EU28 and Other Aisa,"
      "Converting Federal Rep. of Germany to Germany" "Converting Fmr Sudan to Sudan:")
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


##### Vera: filter df_net for categories cereals and iron&steel:
df_net_cereals = df_net.loc[df_net['category'] == '10_cereals', :]
print(df_net_cereals.head())

df_net_ironsteel = df_net.loc[df_net['category'] == '72_iron_and_steel', :]
print(df_net_ironsteel.head())

# preparation for Clustering of Cereals:
# Aggregate per country
df_cluster_cereals = (
    df_net_cereals
    .groupby('country_or_area')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean()
    .reset_index())
# Rescale
scaler_cereals = StandardScaler()
cereals_scaled = scaler_cereals.fit_transform(
    df_cluster_cereals[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']])
print("cereals_scaled type:", type(cereals_scaled))
print("cereals_scaled shape:", cereals_scaled.shape)

# Preparation of Clustering for Iron&Steel:
# Aggregate per country
df_cluster_ironsteel = (
    df_net_ironsteel
    .groupby('country_or_area')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
    .mean()
    .reset_index())
# Rescale
scaler_ironsteel = StandardScaler()
ironsteel_scaled = scaler_ironsteel.fit_transform(
    df_cluster_ironsteel[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']])
print("Iron&Steel_scaled type:", type(ironsteel_scaled))
print("Iron&Steel_scaled shape:", ironsteel_scaled.shape)


##### Thao & Vera: Hierarchical Clustering: Agglomerative

# Agglomerative Clustering for Iron&Steel:

# Linkage matrix
Z = linkage(ironsteel_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=df_cluster_ironsteel['country_or_area'].values,
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

df_cluster_ironsteel['cluster_agglomerative_ironsteel'] = agg.fit_predict(ironsteel_scaled)

print(df_cluster_ironsteel.head())


##### Oliver - Cluster Validity (Plotting z-score heatmap & cluster stability)

ironsteel_features = ['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                      'reexport_ratio', 'reimport_ratio']

plot_heatmap(
    df_cluster=df_cluster_ironsteel,
    scaled=ironsteel_scaled,
    cluster_col="cluster_agglomerative_ironsteel",
    feature_cols=ironsteel_features,
    title="Iron & Steel – Cluster Profiles (Scaled Feature Means)"
)

stability_ironsteel = cluster_stability_repeated(
    X_scaled=ironsteel_scaled,
    n_clusters=3,
    linkage="ward",
    test_size=0.5,
    n_repeats=50,
    seed=123
)

plt.figure(figsize=(8,4))
plt.hist(stability_ironsteel["ARI_B"], bins=12)
plt.title("Iron & Steel – Stability distribution (ARI on B)")
plt.xlabel("ARI (B: assigned-from-A vs full)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

##### Thao & Vera

agglomerative_summary = (
    df_cluster_ironsteel
    .groupby('cluster_agglomerative_ironsteel')[ironsteel_features]
    .mean())

print(agglomerative_summary)

# add the clusters as a column to the iron&steel dataset:
df_net_ironsteel = df_net_ironsteel.merge(
    df_cluster_ironsteel[['country_or_area', 'cluster_agglomerative_ironsteel']],
    on='country_or_area',
    how='left')

save_csv(df_cluster_ironsteel, "df_cluster_ironsteel.csv")
save_csv(df_net, "df_net_new.csv")
save_csv(df_net_ironsteel, "df_net_ironsteel.csv")

# list of clusters:
clusters_ironsteel = (df_cluster_ironsteel.groupby('cluster_agglomerative_ironsteel')['country_or_area']
    .apply(list))
for cluster, countries in clusters_ironsteel.items():
    print(f"Cluster {cluster}:")
    for c in countries:
        print(f"  - {c}")
    print()


# Time Series for Iron&Steel

# sort Data by year:
df_net_ironsteel = df_net_ironsteel.sort_values("year")

# Aggregate one time series per cluster, aggregate within each cluster at each time step
# most common: aggregate with the mean
cluster_ts_ironsteel = (
    df_net_ironsteel
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

##### Oliver - Cluster Validity (Plotting z-score heatmap & cluster stability)

cereals_features = ['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                    'reexport_ratio', 'reimport_ratio']

plot_heatmap(
    df_cluster=df_cluster_cereals,
    scaled=cereals_scaled,
    cluster_col="cluster_agglomerative_cereals",
    feature_cols=cereals_features,
    title="Cereals – Cluster Profiles (Scaled Feature Means)"
)

stability_cereals = cluster_stability_repeated(
    X_scaled=cereals_scaled,
    n_clusters=3,
    linkage="ward",
    test_size=0.5,
    n_repeats=50,
    seed=123
)

plt.figure(figsize=(8,4))
plt.hist(stability_cereals["ARI_B"], bins=12)
plt.title("Cereals – Stability distribution (ARI on B)")
plt.xlabel("ARI (B: assigned-from-A vs full)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


##### Thao & Vera

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
    .groupby('cluster_agglomerative_cereals')[cereals_features]
    .mean()).apply(list)

print(agglomerative_summary)

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

#Forecast for Cluster 0 - Iron & Steel
c0_ironsteel_data = (
    cluster_ts_ironsteel[
        cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 0]
    .sort_values("year")
    .set_index("year"))

h = 5
ts_ntb = c0_ironsteel_data["net_usd"] / (c0_ironsteel_data["Export"] + c0_ironsteel_data["Import"] + 1e-6)
ts_ntb = ts_ntb.sort_index()
model_ntb = ExponentialSmoothing(
    ts_ntb, trend="add", damped_trend=True, seasonal=None).fit()
fc_ntb = model_ntb.forecast(h)
last_year = ts_ntb.index.max()
fc_ntb.index = range(last_year + 1, last_year + h + 1)

plt.figure(figsize=(9,5))
plt.plot(ts_ntb.index, ts_ntb, label="Observed NTB ratio")
plt.plot([ts_ntb.index[-1]] + list(fc_ntb.index),[ts_ntb.iloc[-1]] + list(fc_ntb.values),"--", label="Forecast NTB ratio")
plt.axhline(0, color="black", linewidth=1)
plt.title("Iron & Steel – Cluster 0 – Structural Trade Balance Forecast")
plt.xlabel("Year")
plt.ylabel("NTB ratio")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 2. Level features
plt.figure(figsize=(12,6))
for feature in level_features:
    ts = c0_ironsteel_data[feature].sort_index()
    model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
    fc = model.forecast(h)
    last_year = ts.index.max()
    fc.index = range(last_year + 1, last_year + h + 1)
    line, = plt.plot(ts.index, ts, label=f"{feature} observed")
    plt.plot(
        [last_year] + list(fc.index),
        [ts.iloc[-1]] + list(fc.values),
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Iron & Steel – Cluster 0 – Level indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 3.Ratios SES
ratio_features = ["reexport_ratio", "reimport_ratio"]
plt.figure(figsize=(10,6))
for feature in ratio_features:
    ts = c0_ironsteel_data[feature].sort_index()
    model = SimpleExpSmoothing(ts).fit()
    fc = model.forecast(h)
    last_year = ts.index.max()
    fc.index = range(last_year + 1, last_year + h + 1)
    line, = plt.plot(ts.index, ts, label=f"{feature} observed")
    plt.plot(
        [last_year] + list(fc.index),
        [ts.iloc[-1]] + list(fc.values),
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Iron & Steel – Cluster 0 – Structural ratio forecast")
plt.ylabel("Ratio")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Forecast for Cluster 1 - Iron & Steel
c1_ironsteel_data = cluster_ts_ironsteel[
    cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 1
    ].sort_values("year").set_index("year")
h=5

plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c1_ironsteel_data[feature].sort_index()
    model = ExponentialSmoothing(
        ts, trend="add", damped_trend=True, seasonal=None).fit()
    fc = model.forecast(h)
    last_year = ts.index.max()
    fc.index = range(last_year + 1, last_year + h + 1)
    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(
        [last_year] + list(fc.index),
        [ts.iloc[-1]] + list(fc.values),
        "--",
        color=line.get_color(),
        label=f"Forecast {feature}")

plt.title("Iron & Steel – Cluster 1 – Producer Level Forecast (Scale: 1e9 USD)")
plt.ylabel("Value (USD)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Ratios SES
plt.figure(figsize=(12, 6))
for feature in ratio_features:
    ts = c1_ironsteel_data[feature]
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

#Forecast for Cluster 2 - Iron & Steel
c2_ironsteel_data = cluster_ts_ironsteel[
    cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 2
    ].sort_values("year").set_index("year")

plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c2_ironsteel_data[feature]
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
c0_cereals_data = (
    cluster_ts_cereals[
        cluster_ts_cereals["cluster_agglomerative_cereals"] == 0]
    .sort_values("year")
    .set_index("year")
)
h = 5
plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c0_cereals_data[feature].sort_index()
    model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
    forecast_values = model.forecast(h)

    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")
plt.title("Cereals – Cluster 0 – net_usd Forecast")
plt.xlabel("Year")
plt.ylabel("USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Ratio SES
plt.figure(figsize=(12, 6))
for feature in ratio_features:
    ts = c0_cereals_data[feature]
    model = SimpleExpSmoothing(ts).fit()
    forecast_values = model.forecast(h)

    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Cereals – Cluster 0 – Ratio Forecast")
plt.ylabel("Ratio Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#Forecast for Cluster 1 - Cereals
c1_cereals_data = (
    cluster_ts_cereals[
        cluster_ts_cereals["cluster_agglomerative_cereals"] == 1]
    .sort_values("year")
    .set_index("year"))

h = 5
plt.figure(figsize=(12, 6))
for feature in level_features:
    ts = c1_cereals_data[feature].sort_index()
    model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None).fit()
    forecast_values = model.forecast(h)

    line, = plt.plot(ts.index, ts, label=f"Observed {feature}", linewidth=2)
    plt.plot(range(ts.index.max(), ts.index.max() + h + 1),
             [ts.iloc[-1]] + list(forecast_values),
             linestyle="--", color=line.get_color(), label=f"Forecast {feature}")

plt.title("Cereals – Cluster 1 – net_usd Forecast")
plt.xlabel("Year")
plt.ylabel("USD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
