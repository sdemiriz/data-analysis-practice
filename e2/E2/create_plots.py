import sys
import pandas as pd
import matplotlib.pyplot as plt

# Import both files via Pandas, drop unneeded columns
filename1 = sys.argv[1]
filename2 = sys.argv[2]
file1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes']).drop(axis=1, labels=['lang', 'bytes'])
file2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes']).drop(axis=1, labels=['lang', 'bytes'])

# Sort first file for graphing
file1_sorted = file1.sort_values(['views'], ascending=False)

# Set figure size
plt.figure(figsize=(10, 5))

# Working on first plot, values from first file
plt.subplot(1, 2, 1)
plt.plot(file1_sorted['views'].values, linewidth=2)

# Set title, x label and y label
plt.title('Distribution of Views')
plt.xlabel('Page Index')
plt.ylabel('Page Views')

# Working on second plot, values from joining both files
plt.subplot(1, 2, 2)
file_joined = file1.join(file2, lsuffix='-1', rsuffix='-2')
plt.plot(file_joined['views-1'].values, 
            file_joined['views-2'].values, '.')

# Scale both axes logarithmically
plt.xscale('log')
plt.yscale('log')

# Set title, x label and y label
plt.title('Daily Views')
plt.xlabel('Log Page Index')
plt.ylabel('Log Page Views')

# Show and save as image on disk
#plt.show()
plt.savefig('wikipedia.png')
