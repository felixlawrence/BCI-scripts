import numpy as np
import pandas as pd

from load_full import *

census_data = pd.read_pickle("/home/felix/Data/data_BCI/census_data.pickle")

export_quality = census_data.dbh.unstack(level='CensusID')
export_quality['x'] = census_data.gx.unstack(level='CensusID').bfill(axis=1)[1]
export_quality['y'] = census_data.gx.unstack(level='CensusID').bfill(axis=1)[1]
export_quality['sp']= census_data.sp.unstack(level='CensusID').bfill(axis=1)[1]

export_quality = export_quality.xs(1, level='stemID')
export_quality = export_quality.fillna(-1)

export_quality.to_csv("bci-dbh.csv", index=False, cols=('x', 'y', 3, 4, 5, 6, 7))
export_quality.to_csv("bci-sp.csv",  index=False, cols=('sp',))
