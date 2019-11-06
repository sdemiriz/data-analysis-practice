from datetime import date
from scipy import stats
import pandas as pd
import numpy as np
import sys

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)

def main():
    # Read in json file
    counts = pd.read_json(sys.argv[1], lines=True)

    # Filter years 2012 or 2013, then filter r/canada
    counts = counts[(counts['date'].dt.year == 2012) | (counts['date'].dt.year == 2013)]
    counts = counts[counts['subreddit'] == 'canada']

    # Separate data into weekdays and weekends
    # Adapted from https://stackoverflow.com/questions/19960077/how-to-implement-in-and-not-in-for-pandas-dataframe
    weekends = counts[(counts['date'].apply(date.weekday) == 5) | (counts['date'].apply(date.weekday) == 6)]
    weekdays = counts[~counts.index.isin(weekends.index)]

    # Apply T-test for p-value, normality test, and levene for equal variances
    ttest = stats.ttest_ind(weekends['comment_count'], weekdays['comment_count'])
    initial_weekday_normality=stats.normaltest(weekdays['comment_count'])
    initial_weekend_normality=stats.normaltest(weekends['comment_count'])
    initial_levene = stats.levene(weekdays['comment_count'], weekends['comment_count'])

    # Fix skewedness by log transforming results
    weekdays['comment_count_log'] = np.log(weekdays['comment_count'])
    weekends['comment_count_log'] = np.log(weekends['comment_count'])

    # Apply normality test and levene to transformed values
    transformed_weekday_normality = stats.normaltest(weekdays['comment_count_log'])
    transformed_weekend_normality = stats.normaltest(weekends['comment_count_log'])
    transformed_levene = stats.levene(weekdays['comment_count_log'], weekends['comment_count_log'])

    # Extract year and week from date
    # Adapted from https://stackoverflow.com/questions/23690284/pandas-apply-function-that-returns-multiple-values-to-rows-in-pandas-dataframe
    def extract_from_date(x):
        year, week, _ = date.isocalendar(x)
        return year, week

    # Extract date and aggregate by taking the mean for both parts of the week
    weekdays['iso'] = weekdays['date'].apply(extract_from_date)
    weekdays = weekdays[['comment_count','iso']]
    weekdays = weekdays.groupby('iso').agg('mean')

    weekends['iso'] = weekends['date'].apply(extract_from_date)
    weekends = weekends[['comment_count','iso']]
    weekends = weekends.groupby('iso').agg('mean')

    # Check for normality, equal variance and apply T-test
    weekly_weekday_normality = stats.normaltest(weekdays['comment_count'])
    weekly_weekend_normality = stats.normaltest(weekends['comment_count'])
    weekly_levene = stats.levene(weekdays['comment_count'], weekends['comment_count'])
    weekly_ttest = stats.ttest_ind(weekends['comment_count'], weekdays['comment_count'])

    # Restart from the original data
    weekends = counts[(counts['date'].apply(date.weekday) == 5) | (counts['date'].apply(date.weekday) == 6)]
    weekdays = counts[~counts.index.isin(weekends.index)]

    # Apply U-test
    utest = stats.mannwhitneyu(weekdays['comment_count'], weekends['comment_count'])

    # Print with provided template
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p= ttest.pvalue,
        initial_weekday_normality_p= initial_weekday_normality.pvalue,
        initial_weekend_normality_p= initial_weekend_normality.pvalue,
        initial_levene_p= initial_levene.pvalue,
        transformed_weekday_normality_p= transformed_weekday_normality.pvalue,
        transformed_weekend_normality_p= transformed_weekend_normality.pvalue,
        transformed_levene_p= transformed_levene.pvalue,
        weekly_weekday_normality_p= weekly_weekday_normality.pvalue,
        weekly_weekend_normality_p= weekly_weekend_normality.pvalue,
        weekly_levene_p= weekly_levene.pvalue,
        weekly_ttest_p= weekly_ttest.pvalue,
        utest_p= utest.pvalue,
    ))

if __name__ == '__main__':
    main()