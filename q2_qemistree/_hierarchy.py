# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import biom
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage
from skbio import TreeNode
import numpy as np
from q2_feature_table import merge

from ._collate_fingerprint import collate_fingerprint
from ._match import match_tables
from ._semantics import CSIDirFmt


def _pdist_union(u, v):
    """Compute the number of overlapping elements in two vectors

    Parameters
    ----------
    u, v: array-like
        These vectors are assumed to be of the same length and of ones and
        zeroes.

    Returns
    -------
    float:
        Number of overlapping elements
    """
    return np.double(np.bitwise_or(u, v).sum())


def jaccard_dm(vectors: np.array, attributes: np.array,
               tolerance: float):
    """Jaccard distance with a real-valued attribute comparisons

    Compute the Jaccard distance between each pair `i` and `j` in `vectors`, if
    the difference between `attributes[i]` and `attibutes[j]` is smaller than
    `tolerance` we add one more `True` element to `vectors[i]` and
    `vectors[j]`. In order to perform these computations in an efficient way,
    we first compute a Jaccard distance matrix of the boolean elements in
    `vectors`. We then use this matrix as a baseline for the rest of our
    calculation. In order to do this we calculate to additional matrices, one
    with the pairwise unions of `vectors`, and one with the identity of the
    elements in `attributes`. With these two matrices we can then add a small
    value to the Jaccard distance matrix to account for pairwise-element
    identity.

    We opt to perform the computations in this fashion as the operations can
    all be easily vectorized, as opposed to creating a "custom Jaccard
    distance" for `pdist`.

    Parameters
    ----------
    vectors: array-like 2D
        Matrix of `N` arrays with only zeros and ones.
    attributes: array-like 1D
        Array of `N` elements with floating point values (should have a
        one-to-one correspondance to the elements in `vectors`).
    tolerance: float
        Tolerance to allow for differences between two elements in
        `attributes`.

    Returns
    -------
    array-like
        Condensed distance matrix of adjusted Jaccard distances.
    """
    jsim = 1 - pdist(X=vectors, metric='jaccard')
    union = pdist(X=vectors != 0, metric=_pdist_union)

    # turn the vector into an array of one-element arrays
    mz_diff = pdist(X=attributes.reshape(-1, 1),
                    metric='cityblock')

    mz_diff = (mz_diff <= tolerance).astype(np.int32)

    intersection = jsim * union
    jmod = 1 - ((intersection+mz_diff)/(union+1))

    return jmod


def build_tree(merged_fps: pd.DataFrame,
               merged_fdata: pd.DataFrame,
               mz_tolerance: float) -> TreeNode:
    """Create a tree for the molecules described by the parameters

    Parameters
    ----------
    merged_fps: pd.DataFrame
        Predicted fingerprints for a set of molecules, indexed by the molecule.
    merged_fdata: pd.DataFrame
        Additional metadata for the molecules, indexed by the molecule and
        having an overlap with `merged_fps`. Should include a column named 'row
        m/z`.
    mz_tolerance: float
        Allowed tolerance when comparing the m/z between two molecules.

    Returns
    -------
    TreeNode:
        Tree representing the relatedness of the molecules.
    """

    # sort these elements to be in the same order
    merged_fdata = merged_fdata.loc[merged_fps.index]
    merged_fdata['row m/z'] = pd.to_numeric(merged_fdata['row m/z'])

    distmat = jaccard_dm(merged_fps.values, merged_fdata.values, mz_tolerance)

    distsq = squareform(distmat, checks=False)
    linkage_matrix = linkage(distsq, method='average')
    tree = TreeNode.from_linkage_matrix(linkage_matrix,
                                        merged_fps.index.tolist())
    return tree


def merge_relabel(fps: pd.DataFrame, fts: pd.DataFrame, fdata: pd.DataFrame):
    '''
    This function merges fingerprints, feature tables and feature data tables
    from multiple datasets.
    '''
    # the length of fps, fts and fdata should be the same (checked in
    # make_hierarchy)
    for i in range(len(fps)):
        n = str(i+1)

        fdata[i].index = ['table' + n + '_' + fid for fid in fdata[i].index]
        fps[i].index = ['table' + n + '_' + fid for fid in fps[i].index]
        fts[i].index = ['table' + n + '_' + fid for fid in fts[i].index]

        fts[i] = biom.table.Table(data=fts[i].values,
                                  observation_ids=fts[i].index.astype(str),
                                  sample_ids=fts[i].columns.astype(str))

    merged_fps = pd.concat(fps)
    merged_fps.index.name = '#featureID'
    merged_fdata = pd.concat(fdata)
    merged_fdata.index.name = '#featureID'
    merged_ftable = merge(fts, overlap_method='error_on_overlapping_sample')
    merged_ftable.table_id = '#featureID'

    return merged_fps, merged_ftable, merged_fdata


def make_hierarchy(csi_results: CSIDirFmt,
                   feature_tables: biom.Table,
                   feature_data: pd.DataFrame,
                   mz_tolerance: float = 0.01,
                   qc_properties: bool = False) -> (TreeNode, biom.Table,
                                                    pd.DataFrame):
    '''
    This function generates a hierarchy of mass-spec features based on
    predicted chemical fingerprints. It filters the feature table to
    retain only the features with fingerprints and relables each feature with
    a hash (MD5) of its binary fingerprint vector.

    Parameters
    ----------
    csi_results : CSIDirFmt
        one or more CSI:FingerID output folder
    feature_table : biom.Table
        one or more feature tables with mass-spec feature intensity per sample
    feature_data : pd.DataFrame
        metadata (row ID, row m/z) about features in the feature table
    mz_tolerance : float, optional
        maximum allowable tolerance in m/z of parent ions
    qc_properties : bool, default False
        flag to filter molecular properties to keep only PUBCHEM fingerprints

    Raises
    ------
    ValueError
        If number of feature_tables don't match the number of csi_results
        If number of feature_tables don't match the number of feature_data sets
        If any feature table in empty
        If any collated fingerprint table is empty

    Returns
    -------
    skbio.TreeNode
        a tree of relatedness of molecules
    biom.Table
        matched feature table that is filtered to contain only the
        features present in the tree
    pd.DataFrame
        matched feature data
    '''

    fps, fts, fds = [], [], []

    if len(feature_tables) != len(csi_results):
        raise ValueError("The feature tables and CSI results should have a "
                         "one-to-one correspondance.")
    if len(feature_tables) != len(feature_data):
        raise ValueError("The feature tables and feature data should have a "
                         "one-to-one correspondance.")

    for f_table, f_data, csi in zip(feature_tables, feature_data, csi_results):
        if f_table.is_empty():
            raise ValueError("Cannot have an empty feature table")

        fingerprints = collate_fingerprint(csi, qc_properties)
        bin_fps, matched_ftable, matched_fdata = match_tables(fingerprints,
                                                              f_table,
                                                              f_data)

        fps.append(bin_fps)
        fts.append(matched_ftable)
        fds.append(matched_fdata)

    merged_fps, merged_ftable, merged_fdata = merge_relabel(fps, fts, fds)
    tree = build_tree(merged_fps, merged_fdata, mz_tolerance)

    return tree, merged_ftable, merged_fdata
