from pandas import read_csv, DataFrame
import pandas as pd
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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
sns.scatterplot(x='net_imports', y='net_exports', hue='country_or_area', data=df_net)
plt.title('Imports vs. Exports Colored by Country')
plt.xlabel('net_imports')
plt.ylabel('net_export')
plt.legend(title='Country')

plt.show()