from pandas import read_csv
import pandas as pd

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
    lambda x: x.sample(frac=0.1))
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


# Oliver - Plot Albanian cereal production over the years
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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