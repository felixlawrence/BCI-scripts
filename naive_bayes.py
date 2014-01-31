import numpy as np
import pandas as pd

from load_full import stack_w_censusid

class NaiveBayes(object):
    """ Histogram-based Naive Bayes treatment for census data."""
    def __init__(self, training_data,
                 dbh_bins=np.arange(10,90)-0.5,
                 growth_bins=np.arange(-0.4, 2.3, 0.2)-0.1):

        t_data = training_data
        self.dbh_bins = dbh_bins
        self.growth_bins = growth_bins

        # Get pts corresponding to each classification value
        surv = t_data.survives.fillna(False)
        dies = t_data.dies.fillna(False)
        self.prior_surv = 1. * surv.sum() / (surv.sum() + dies.sum())
        self.prior_dies = 1. * dies.sum() / (surv.sum() + dies.sum())

        dbh = t_data.dbh
        surv_dbhs = np.histogram(dbh[surv].dropna(),
                                 bins=dbh_bins, normed=True)[0]
        dies_dbhs = np.histogram(dbh[dies].dropna(),
                                 bins=dbh_bins, normed=True)[0]

        growth = t_data.prev_growth
        surv_growths = np.histogram(growth[surv].dropna(), bins=growth_bins,
                                    normed=True)[0]
        dies_growths = np.histogram(growth[dies].dropna(), bins=growth_bins,
                                    normed=True)[0]

        self.surv_mat = np.outer(surv_dbhs, surv_growths) * self.prior_surv
        self.dies_mat = np.outer(dies_dbhs, dies_growths) * self.prior_dies

        # Normalise survival/dying probabilities to add to 1 (for thresholding)
        norm_mat = self.surv_mat + self.dies_mat
        self.surv_mat /= norm_mat
        self.dies_mat /= norm_mat


    def predict(self, test_data, thres=0.5):
        """ Predict whether each sampled tree will survive 5 years.

            Return True if it will survive, False if it will die.
        """
        # Put bin edges are along a different axis to dbh, for broadcasting
        dbh_lb = self.dbh_bins[:-1].reshape(1, -1)      # Lower bin edges
        dbh_ub = self.dbh_bins[1:].reshape(1, -1)       # Upper bin edges
        growth_lb = self.growth_bins[:-1].reshape(1, -1)
        growth_ub = self.growth_bins[1:].reshape(1,-1)

        #print len(test_data.dbh), len(test_data.prev_growth)
        # Drop samples that don't fit a bin
        data = test_data.ix[(test_data.dbh.values >= dbh_lb[0, 0]) &
                            (test_data.dbh.values < dbh_ub[0, -1]) &
                            (test_data.prev_growth.values >= growth_lb[0, 0]) &
                            (test_data.prev_growth.values < growth_ub[0, -1])]

        # Find the dbh bin that each sample is in
        dbh = data.dbh.values.reshape(-1, 1)

        is_dbh_in_bin = (dbh >= dbh_lb) & (dbh < dbh_ub)
        # Which bin is this dbh in?
        dbh_bin_idx = is_dbh_in_bin.nonzero()[1]

        # Now for prev_growth:
        growth = data.prev_growth.values.reshape(-1, 1)

        is_growth_in_bin = (growth >= growth_lb) & (growth < growth_ub)
        growth_bin_idx = is_growth_in_bin.nonzero()[1]

        prediction_by_bin = (self.surv_mat >= thres)

        #print dbh_bin_idx, growth_bin_idx
        #print dbh_bin_idx.shape, growth_bin_idx.shape

        data['pred_surv'] = prediction_by_bin[dbh_bin_idx, growth_bin_idx]
        return data
