import pandas as pd
import numpy as np
import random
import time

# Get sort implementations
from implementations import all_implementations

# Preset parameters
array_size = 10000
max_value = 1000
num_tests = 100

# Get random values
random_values = np.random.randint(max_value, size = array_size)
values = pd.DataFrame()

# Run and time sorts, get corresponding durations
for sort in all_implementations:
    duration = []
    for i in range(num_tests):
        
        st = time.time()
        res = sort(random_values)
        en = time.time()
    
        duration.append(en-st)
    values[sort.__name__] = pd.Series(duration)

# Export data as csv
values.to_csv('data.csv', index=False)