from pandas import read_csv

df = read_csv("case2.csv", sep=";")
print(df.head())