from pandas import read_csv

df = read_csv("case2.csv", sep=";")
df = df.drop(columns=["commodity", "comm_code"])
print(df.head())
print(df.columns)
df = df[~df["country_or_area"].isin([
    "EU-28",
    "So. African Customs Union"])]
df["country_or_area"] = df["country_or_area"].replace(
    "Fmr Fed. Rep. of Germany", "Germany")
