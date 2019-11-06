from statsmodels.stats.multicomp import pairwise_tukeyhsd as tukey
import pandas as pd

# Read in data.csv
data = pd.read_csv('data.csv')

# From lecture notes: Statistical Tests: Post Hoc Analysis
x_melt = pd.melt(data)
posthoc = tukey(x_melt['value'], x_melt['variable'], alpha=0.05)

# Print results
print("Data Means \n", data.mean(), "\n")
print(posthoc)