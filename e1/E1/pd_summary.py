import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

print("Row with lowest total precipitation:")
print(totals.sum(axis = 1).idxmin()) # Row with lowest number

print("Average precipitation in each month:")
print(totals.sum(axis=0)/counts.sum(axis=0)) # Average precip. per month

print("Average precipitation in each city:")
print(totals.sum(axis=1)/counts.sum(axis=1)) # Average precip. per city