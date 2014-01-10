from nose.tools import *

import pandas as pd
import numpy as np

from load_full import *

census_data_pickle = pd.read_pickle('tests/fixtures/census_data.pickle')


def test_calc_species_abundance():
    census_data = census_data_pickle.copy()
    abund = calc_species_abundance(census_data)
    assert_equal(326,       len(abund))
    assert_equal(7108,      abund['acaldi'])


def test_calc_dt_days():
    census_data = census_data_pickle.copy()
    # If the column already exists then we're not really testing it!
    assert_not_in('dt_days', census_data.columns)
    data_subset = census_data.ix[census_data['sp'] == 'gar2in']
    calc_dt_days(data_subset)
    assert_in('dt_days', data_subset.columns)
    assert_equal(25374,     len(data_subset.dt_days.dropna()))
    assert_equal(1477,      data_subset.dt_days[1,212,1])


def test_load_full():
    census_data = load_all_censuses()
    assert_equal(2242742,   len(census_data))       # full data
    # assert_equal(1860313,   len(census_data))       # drop nan censusID
    assert_equal(9194,      len(census_data.dropna()))
    assert_equal(258723,    ((census_data.sp == 'hybapr') *
                             (census_data.status == 'A')).sum())
    assert_equal(np.dtype('<M8[ns]'), census_data['pddate'].dtype)
    assert_equal(1860136,   len(census_data['pddate'].dropna()))
