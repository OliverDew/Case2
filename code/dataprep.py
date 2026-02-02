from typing import Optional

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas import read_csv, DataFrame

##### Oliver - Read CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)

##### Oliver Save CSV function
def save_csv(dataframe: DataFrame, name_of_csv: str):
    dataframe.to_csv(CSV_DIR / name_of_csv, index=False)

df = read_csv(CSV_DIR / "case2.csv", sep=";")

print("Initial Dataset:")
print(df.info())

##### Oliver - Reusable dataprep function (mainly for k-means.py)
df_net: Optional[DataFrame] = None
cereals_scaled = None
ironsteel_scaled = None
scaler_cereals = None
scaler_ironsteel = None
df_net_ironsteel: Optional[DataFrame] = None
df_net_cereals: Optional[DataFrame] = None
df_cluster_ironsteel: Optional[DataFrame] = None
df_cluster_cereals: Optional[DataFrame] = None
def dataprep():
    # Using global to make sure that the funciton uses the variables on root-level
    global df
    global df_net
    global cereals_scaled, ironsteel_scaled
    global scaler_cereals, scaler_ironsteel
    global df_net_ironsteel, df_net_cereals
    global df_cluster_ironsteel, df_cluster_cereals

    ##### Oliver - Delete all CSVs except case2 + sample on each run
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