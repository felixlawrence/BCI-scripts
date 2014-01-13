import numpy as np
import pandas as pd

data_path = "/home/felix/Data/data_BCI/rawdata/"
file_idxs = [i+1 for i in xrange(7)]


def refresh_data_pickle():
    print "loading censuses"
    census_data = load_all_censuses()
    print "calc_dt_days"
    calc_dt_days(census_data)
    print "calc_growth"
    calc_growth(census_data)
    print "saving pickle..."
    census_data.to_pickle("/home/felix/Data/data_BCI/census_data.pickle")
    return census_data


def load_one_census(i):
    file_name = "bci.full%d.csv" % i
    file_path = data_path + "50ha/" + file_name
    data = pd.read_csv(file_path)

    # We're going to roll our own census id because the one provided
    # has lots of NaNs :-(
    data.rename(columns={'CensusID': 'origCensusID'}, inplace=True)

    # Drop data from trees that don't exist yet
    data = data.drop(data.index[data['status'] == 'P'])

    # Convert dates to datetime
    data['pddate'] = pd.to_datetime(data.ExactDate)

    # PANDAS BUG WORKAROUND:
    # nan indexes are not supported properly
    # data = data[np.logical_not(data.origCensusID.isnull())]
    assert (data.stemID != 0).all()
    data.stemID.fillna(0, inplace=True)
    # idx = pd.MultiIndex.from_arrays((data.treeID, data.stemID),
    #                                 names=('treeID', 'stemID'))
    # data.index = idx
    data.set_index(['treeID', 'stemID'], inplace=True)

    # data = data.reindex(index=('treeID', 'stemID'))
    assert not data.index.has_duplicates

    return data


def load_all_censuses():
    all_data_list = [load_one_census(i) for i in file_idxs]
    all_data = pd.concat(all_data_list, keys=file_idxs)
    all_data.index.names[0] = 'CensusID'

    return all_data


def load_seed_rain():
    file_path = data_path + "seedrain/" \
        + "Seeds_Recruits_by_Sp_Yr_Station_Includes_Metadata.csv"
    seed_rain = pd.read_csv(file_path, index_col='year')#, index_col=('trap', 'sp', 'year'))
    return seed_rain


def load_trap_locations():
    file_path = data_path + "seedrain/" + "trap_locations.csv"
    trap_locations = pd.read_csv(file_path, index_col='TRAP')
    trap_locations.columns = trap_locations.columns.map(str.lower)
    trap_locations = trap_locations.drop(('q20x', 'q20y', 'q5x', 'q5y'))
    return trap_locations


def calc_species_abundance(census_data = None):
    """ Count the observations of live plants of each species.

        N.B. if a plant is alive across three censuses, then it's
        counted three times.
    """
    if census_data is None:
        census_data = load_all_censuses()

    alive_samples = census_data[census_data['status'] == 'A']
    abundance_list = alive_samples.groupby('sp').size()
    return abundance_list


def calc_dt_days(census_data):
    dates = census_data.pddate.unstack(level='CensusID')

    # Bug in pandas: https://github.com/pydata/pandas/issues/4533
    dt = dates.apply(pd.Series.diff, axis=1)
    # Annoyance in numpy: http://stackoverflow.com/questions/18215317/extracting-days-from-a-numpy-timedelta64-value
    dt_days = dt/np.timedelta64(1, 'D')

    # store t_{i+1} - t_i with data point i
    dt_days = dt_days.iloc[:,1:]
    dt_days.columns = dt.columns[:-1]

    census_data['dt_days'] = stack_w_censusid(dt_days)


def calc_growth(census_data):
    """ Calculate future dbh growth rate [mm/day]"""
    dbhs = census_data.dbh.unstack(level='CensusID')

    ddbhs = dbhs.T.diff().T         # differentiate along censusID
    ddbhs = ddbhs.iloc[:,1:]
    ddbhs.columns = dbhs.columns[:-1]

    census_data['growth'] = stack_w_censusid(ddbhs) / census_data.dt_days


def calc_death(census_data):
    is_alive = (census_data.status == 'A').unstack(level='CensusID')

    # Is it alive now and in the following census?
    survives = is_alive.iloc[:,:-1] & is_alive.iloc[:,1:].values

    dies     = is_alive.iloc[:,:-1] & np.logical_not(is_alive.iloc[:,1:].values)

    census_data['survives'] = stack_w_censusid(survives)
    census_data['dies']     = stack_w_censusid(dies)


def stack_w_censusid(df):
    s = df.stack()
        # Pandas sometimes forgets the name here (haven't got a unit test for this)
    if not 'CensusID' in s.index.names:
        print "forgot name of CensusID", s.index.names
        s.index.names[-1] = 'CensusID'

    return s.reorder_levels(['CensusID','treeID','stemID'])
