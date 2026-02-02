from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pathlib import Path

import dataprep

##### Oliver - Read CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)

##### Oliver Save CSV function
def save_csv(dataframe: DataFrame, name_of_csv: str):
    dataframe.to_csv(CSV_DIR / name_of_csv, index=False)


##### Oliver - Run dataprep function and get variables
dataprep.dataprep()
df_net = dataprep.df_net
df_net_ironsteel = dataprep.df_net_ironsteel
df_net_cereals = dataprep.df_net_cereals
ironsteel_scaled = dataprep.ironsteel_scaled
cereals_scaled = dataprep.cereals_scaled
df_cluster_ironsteel = dataprep.df_cluster_ironsteel
df_cluster_cereals = dataprep.df_cluster_cereals

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
