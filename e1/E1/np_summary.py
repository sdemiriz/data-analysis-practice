import numpy as np

data = np.load('monthdata.npz')

totals = data['totals']
counts = data['counts']

print("Row with lowest total precipitation:")
print(totals.sum(axis = 1).argmin()) # Row with lowest number

print("Average precipitation in each month:")
print(totals.sum(axis=0)/counts.sum(axis=0)) # Average precip. per month

print("Average precipitation in each city:")
print(totals.sum(axis=1)/counts.sum(axis=1)) # Average precip. per city

print("Quarterly precipitation totals:")
print(totals.reshape(-1,3).sum(axis=1).reshape(-1,4)) # Quarterly precip. sums