from pandas import read_csv
import pandas as pd

df = read_csv("case2.csv", sep=";")
df = df.drop(columns=["commodity", "comm_code"])
print(df.head())
print(df.columns)
df = df[~df["country_or_area"].isin([
    "EU-28",
    "So. African Customs Union"])]
df["country_or_area"] = df["country_or_area"].replace(
    "Fmr Fed. Rep. of Germany", "Germany")

df = df.groupby('country_or_area', group_keys=False).apply(
    lambda x: x.sample(frac=0.1))
df = df.reset_index(drop=True)
df.to_csv("case2_sampled.csv", index=False)

df.info()
print(df.info())
df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].median())
print(df.info())

df_dummy = (pd.get_dummies(df, columns = ['category', 'flow'], prefix_sep='=', dummy_na=False, dtype='int'))
print(df_dummy.head())
df_dummy.to_csv("dummy.csv", index=False)