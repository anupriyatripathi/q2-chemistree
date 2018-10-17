# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
from biom.table import Table
import pandas as pd
import numpy as np
from q2_chemistree import make_hierarchy


class test_hierarchy(TestCase):
    def setUp(self):
        tablefp = Table({}, [], [])
        self.emptyin = tablefp

        table = pd.DataFrame({'1': [1, 1, 1], '2': [0, 1, 1],
                              '3': [0, 0, 1], '4': [0, 0, 0]})
        nptable = np.asarray(table)
        tablefp = Table(data=nptable, observation_ids=table.index,
                        sample_ids=table.columns)
        self.goodin = tablefp

        self.bigthresh = 1.5
        self.smallthresh = -0.5
        self.goodthresh = 0.5

    def test_emptyFingerprint(self):
        with self.assertRaises(ValueError):
            make_hierarchy(self.emptyin, prob_threshold=self.goodthresh)

    def test_thresholdError(self):
        with self.assertRaises(ValueError):
            make_hierarchy(self.goodin, prob_threshold=self.bigthresh)
        with self.assertRaises(ValueError):
            make_hierarchy(self.goodin, prob_threshold=self.smallthresh)

    def test_tipMatch(self):
        treeout = make_hierarchy(self.goodin, prob_threshold=self.goodthresh)
        tip_names = {node.name for node in treeout.tips()}
        self.assertEqual(tip_names, set(self.goodin._observation_ids))


if __name__ == '__main__':
    main()
