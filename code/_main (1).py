from pandas import read_csv, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

##### Oliver - Delete all CSVs except case2 and sample on each run
for file in CSV_DIR.glob("*.csv"):
    if file.name not in {"case2.csv", "case2_sampled.csv"}:
        file.unlink()

# Paula - Clean and Remove Columns
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

# Vera: filter df_net for categories cereals and iron&steel:
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


#Thao & Vera: Hierarchical Clustering: Agglomerative

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

agglomerative_summary = (
    df_cluster_ironsteel
    .groupby('cluster_agglomerative_ironsteel')[['Export','Import','Re-Export', 'Re-Import', 'net_usd',
                                 'reexport_ratio', 'reimport_ratio']]
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
    .sort_values("year"))
c0_ironsteel_data["year"] = pd.to_datetime(c0_ironsteel_data["year"], format="%Y")
c0_ironsteel_data = c0_ironsteel_data.set_index("year")
c0_ironsteel_data.index.freq = 'YS-JAN' # type: ignore

ts_ntb = (c0_ironsteel_data["net_usd"]
    / (c0_ironsteel_data["Export"] + c0_ironsteel_data["Import"] + 1e-6))

ts_ntb = ts_ntb.asfreq("YS")  # yearly start frequency

model_ntb = ExponentialSmoothing(
    ts_ntb, damped_trend=False, seasonal=None).fit()
fc_ntb = model_ntb.forecast(5)

plt.figure(figsize=(9,5))
plt.plot(ts_ntb, label="Observed NTB ratio")
fc_plot = pd.concat([ts_ntb.iloc[-1:], fc_ntb])
plt.plot(fc_plot, "--", label="Forecast NTB ratio")
plt.axhline(0, color="black", linewidth=1)
plt.title("Iron & Steel – Cluster 0 – Structural Trade Balance Forecast")
plt.xlabel("Year")
plt.ylabel("NTB ratio")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 2. Level features
ts = c0_ironsteel_data[feature].copy

plt.figure(figsize=(12,6))

for feature in level_features:
    ts = c0_ironsteel_data[feature]
    model = ExponentialSmoothing(
        ts,
        trend="add",
        damped_trend=False,
        seasonal=None
    ).fit()

    fc = model.forecast(5)

    # plot observed
    line, = plt.plot(ts, label=f"{feature} observed")

    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
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
plt.figure(figsize=(12,6))

for feature in ratio_features:
    ts = c0_ironsteel_data[feature].sort_index()
    model = ExponentialSmoothing(
        ts,
        damped_trend=False,
        seasonal=None
    ).fit()

    fc = model.forecast(5)
    line, = plt.plot(ts, label=f"{feature} observed")
    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Iron & Steel – Cluster 0 – Ratio indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Forecast for Cluster 1 - Iron & Steel
c1_ironsteel_data = (
    cluster_ts_ironsteel[
        cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 1]
    .sort_values("year"))
c1_ironsteel_data["year"] = pd.to_datetime(c1_ironsteel_data["year"], format="%Y")
c1_ironsteel_data = c1_ironsteel_data.set_index("year")
c1_ironsteel_data.index.freq = 'YS-JAN' # type: ignore

ts = c1_ironsteel_data[feature].copy

plt.figure(figsize=(12,6))

for feature in level_features:
    ts = c1_ironsteel_data[feature]
    model = ExponentialSmoothing(
        ts,
        trend="add",
        damped_trend=False,
        seasonal=None
    ).fit()

    fc = model.forecast(5)

    line, = plt.plot(ts, label=f"{feature} observed")

    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Iron & Steel – Cluster 1 – Level indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Ratios SES
plt.figure(figsize=(12,6))

for feature in ratio_features:
    ts = c1_ironsteel_data[feature].sort_index()
    model = ExponentialSmoothing(
        ts,
        damped_trend=False,
        seasonal=None
    ).fit()

    fc = model.forecast(5)

    line, = plt.plot(ts, label=f"{feature} observed")

    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Iron & Steel – Cluster 1 – Ratio indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Forecast for Cluster 2 - Iron & Steel
c2_ironsteel_data = (
    cluster_ts_ironsteel[
        cluster_ts_ironsteel["cluster_agglomerative_ironsteel"] == 2]
    .sort_values("year"))
c2_ironsteel_data["year"] = pd.to_datetime(c2_ironsteel_data["year"], format="%Y")
c2_ironsteel_data = c2_ironsteel_data.set_index("year")
c2_ironsteel_data.index.freq = 'YS-JAN' # type: ignore

ts = c2_ironsteel_data[feature].copy

plt.figure(figsize=(12,6))

for feature in level_features:
    ts = c2_ironsteel_data[feature]
    model = ExponentialSmoothing(
        ts,
        trend="add",
        damped_trend=False,
        seasonal=None
    ).fit()
    fc = model.forecast(5)
    line, = plt.plot(ts, label=f"{feature} observed")
    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Iron & Steel – Cluster 2 – Level indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Forecast for Cluster 0 – Cereals
c0_cereals_data = (
    cluster_ts_cereals[
        cluster_ts_cereals["cluster_agglomerative_cereals"] == 0]
    .sort_values("year"))
c0_cereals_data["year"] = pd.to_datetime(c0_cereals_data["year"], format="%Y")
c0_cereals_data = c0_cereals_data.set_index("year")
c0_cereals_data.index.freq = 'YS-JAN' # type: ignore

ts = c0_cereals_data[feature].copy

plt.figure(figsize=(12,6))

for feature in level_features:
    ts = c0_cereals_data[feature]
    model = ExponentialSmoothing(
        ts,
        trend="add",
        damped_trend=False,
        seasonal=None
    ).fit()

    fc = model.forecast(5)
    line, = plt.plot(ts, label=f"{feature} observed")
    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Cereals – Cluster 0 – Level indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Ratio SES
plt.figure(figsize=(12,6))

for feature in ratio_features:
    ts = c0_cereals_data[feature].sort_index()
    model = ExponentialSmoothing(
        ts,
        damped_trend=False,
        seasonal=None
    ).fit()

    fc = model.forecast(5)

    line, = plt.plot(ts, label=f"{feature} observed")

    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Cereals – Cluster 0 – Ratio indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#Forecast for Cluster 1 - Cereals
c1_cereals_data = (
    cluster_ts_cereals[
        cluster_ts_cereals["cluster_agglomerative_cereals"] == 1]
    .sort_values("year"))
c1_cereals_data["year"] = pd.to_datetime(c1_cereals_data["year"], format="%Y")
c1_cereals_data = c1_cereals_data.set_index("year")
c1_cereals_data.index.freq = 'YS-JAN' # type: ignore

ts = c1_cereals_data[feature].copy

plt.figure(figsize=(12,6))

for feature in level_features:
    ts = c1_cereals_data[feature]
    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal=None
    ).fit()
    fc = model.forecast(5)
    line, = plt.plot(ts, label=f"{feature} observed")
    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Cereals – Cluster 1 – Level indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Ratio SES
plt.figure(figsize=(12,6))

for feature in ratio_features:
    ts = c1_cereals_data[feature].sort_index()
    model = ExponentialSmoothing(
        ts,
        seasonal=None
    ).fit()

    fc = model.forecast(5)

    line, = plt.plot(ts, label=f"{feature} observed")

    fc_plot = pd.concat([ts.iloc[-1:], fc])
    plt.plot(
        fc_plot,
        "--",
        color=line.get_color(),
        label=f"{feature} forecast")

plt.title("Cereals – Cluster 1 – Ratio indicators forecast")
plt.ylabel("USD")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)