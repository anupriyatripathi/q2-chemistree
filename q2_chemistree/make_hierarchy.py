import biom
from scipy.cluster import hierarchy
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from skbio.tree import TreeNode


def make_hierarchy(tablefp, threshold):
    '''
    This function makes a tree of relatedness between mass-spectrometry features
    using molecular substructure information.

    Parameters
    ----------
    tablefp : biom object
        biom object containing mass-spec feature IDs (in rows) and molecular
        substructure IDs (in columns). Values are probabilities that a feature
        contains a particular substructure.
    threshold : float
            probability value below which a molecular substructure is considered
            absent from a feature
    Returns:
    ----------
    tree : scikit-bio TreeNode object
        a tree of relatedness of molecules
    '''

    table = tablefp.to_dataframe()
    if table.shape == (0, 0):
        raise ValueError("Cannot have empty fingerprint table")
    if not 0 <= threshold <= 1:
        raise ValueError("Probability threshold is not in [0,1]")
    for col in table:
        table[col] = [1 if val > threshold else 0 for val in table[col]]
    distmat = pairwise_distances(X=table, Y=None, metric='jaccard')
    distsq = squareform(distmat)
    linkage_matrix = linkage(distsq, method='average')
    tree = TreeNode.from_linkage_matrix(linkage_matrix, list(table.index))
    return tree
