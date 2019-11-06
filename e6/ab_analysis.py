import scipy.stats as st
import pandas as pd
import numpy as np
import sys

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)

def main():
    searchdata_file = sys.argv[1]
    searchdata = pd.read_json(searchdata_file, orient='records', lines= True)

    # Get odd and even uid for all users
    odds_all = searchdata[searchdata['uid'] % 2 == 1]                   # New search box
    evens_all = searchdata[searchdata['uid'] % 2 == 0]                  # Old search box

    # Get all people who searched and people who didn't
    oddsSearch_all = odds_all[odds_all['search_count'] > 0]             # Used search
    oddsNoSearch_all = odds_all[odds_all['search_count'] == 0]          # Unused search

    evensSearch_all = evens_all[evens_all['search_count'] > 0]          # Used search
    evensNoSearch_all = evens_all[evens_all['search_count'] == 0]       # Unused search

    # Form data from all into contingency table using counts
    table_all = [[oddsSearch_all['uid'].count(), oddsNoSearch_all['uid'].count()], \
                 [evensSearch_all['uid'].count(), evensNoSearch_all['uid'].count()]]

    # Return values for all people chi2 analysis and Whitney Test p-value
    chi2_all, p_all, dof_all, expected_all = st.chi2_contingency(table_all)
    whitneyP_all = st.mannwhitneyu(odds_all['search_count'], evens_all['search_count']).pvalue

    # Isolate instructors from original dataset
    search_instr = searchdata[searchdata['is_instructor'] == True]

    # Find instructors with odd and even uid's
    odds_instr = search_instr[search_instr['uid'] % 2 == 1]             # New search box
    evens_instr = search_instr[search_instr['uid'] % 2 == 0]            # Old search box

    # Get instructors who searched and instructors who didn't
    oddsSearch_instr = odds_instr[odds_instr['search_count'] > 0]       # Used search
    oddsNoSearch_instr = odds_instr[odds_instr['search_count'] == 0]    # Unused search

    evensSearch_instr = evens_instr[evens_instr['search_count'] > 0]    # Used search
    evensNoSearch_instr = evens_instr[evens_instr['search_count'] == 0] # Unused search

    # Form data from instructors into contingency table using counts
    table_instr = [[oddsSearch_instr['uid'].count(), oddsNoSearch_instr['uid'].count()], \
                   [evensSearch_instr['uid'].count(), evensNoSearch_instr['uid'].count()]]

    # Return values for instructor chi2 analysis and Whitney Test p-value
    chi2_instr, p_instr, dof_instr, expected_instr = st.chi2_contingency(table_instr)
    whitneyP_instr = st.mannwhitneyu(odds_instr['search_count'], evens_instr['search_count']).pvalue

    # Print results
    print(OUTPUT_TEMPLATE.format(
        more_users_p = p_all,
        more_searches_p = whitneyP_all,
        more_instr_p = p_instr,
        more_instr_searches_p = whitneyP_instr,
    ))

if __name__ == '__main__':
    main()