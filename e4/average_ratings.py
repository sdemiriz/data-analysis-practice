import difflib as dl
import pandas as pd
import numpy as np
import sys

# Get filenames for list and ratings
movie_list_file = sys.argv[1]
movie_ratings_file = sys.argv[2]

# Open stream, read lines, close stream
f = open(movie_list_file)
read = f.readlines()
f.close()

# Clean strings, form dataframe from list
movie_list = pd.DataFrame(read, columns= ['movie'])
movie_list = movie_list.replace(r'\n', '', regex= True)

# Read movie rating data
movie_ratings = pd.read_csv(movie_ratings_file)

# Define a ufunc for getting close matches
# Idea from https://stackoverflow.com/questions/46083151/python-3-6-pandas-difflib-get-close-matches-to-filter-a-dataframe-with-user-inpu
def get_close_ufunc(rating):
    match = dl.get_close_matches(rating, movie_list['movie'])
    if len(match) != 0:
        return match[0]
    return None

# Apply funtion to transform tidy up movie titles
movie_ratings['title'] = movie_ratings['title'].apply(get_close_ufunc)

# Aggregate by title, summing the ratings, the averaging the values
agg_func = {'title': 'count', 'rating': 'sum'}
agg_ratings = movie_ratings.groupby(movie_ratings['title']).aggregate(agg_func)
agg_ratings['rating'] = agg_ratings['rating'] / agg_ratings['title']
agg_ratings['rating'] = agg_ratings['rating'].round(2)

# Output to .csv file from last argument
agg_ratings['rating'].to_csv(sys.argv[3], header= True)