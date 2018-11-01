# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from ._fingerprint import fingerprint
from ._hierarchy import make_hierarchy
from ._match import match_table
from ._collate_fingerprint import collate_fingerprint
from ._semantics import (MassSpectrometryFeatures, MGFDirFmt,
                         CSIFingerprintFolder, CSIDirFmt)

from qiime2.plugin import Plugin, Str, Range, Choices, Float, Int
from q2_types.feature_table import FeatureTable, Frequency
from q2_types.tree import Phylogeny, Rooted

plugin = Plugin(
    name='chemistree',
    version='0.0.0',
    website='https://github.com/biocore/q2-chemistree',
    package='q2_chemistree',
    description='Hierarchical orderings for mass spectrometry data',
    short_description='Plugin for exploring chemical diversity.',
)

# type registration
plugin.register_views(MGFDirFmt)
plugin.register_semantic_types(MassSpectrometryFeatures)
plugin.register_semantic_type_to_format(MassSpectrometryFeatures,
                                        artifact_format=MGFDirFmt)

plugin.register_views(CSIDirFmt)
plugin.register_semantic_types(CSIFingerprintFolder)
plugin.register_semantic_type_to_format(CSIFingerprintFolder,
                                        artifact_format=CSIDirFmt)

PARAMS = {
    'database': Str % Choices(['all', 'pubchem']),
    'sirius_path': Str,
    'profile': Str % Choices(['qtof', 'orbitrap', 'fticr']),
    'fingerid_db': Str % Choices(['all', 'pubchem', 'bio', 'kegg', 'hmdb']),
    'ppm_max': Int % Range(0, 30, inclusive_end=True),
    'n_jobs': Int % Range(1, None),
    'num_candidates': Int % Range(5, 100, inclusive_end=True),
    'tree_timeout': Int % Range(600, 3000, inclusive_end=True),
    'maxmz': Int % Range(100, 850, inclusive_end=True),
    'zodiac_threshold': Float % Range(0, 1, inclusive_end=True),
}

PARAMS_DESC = {
    'database': 'search formulas in given database',
    'sirius_path': 'path to Sirius executable',
    'ppm_max': 'allowed parts per million tolerance for decomposing masses',
    'profile': 'configuration profile for mass-spec platform used',
    'n_jobs': 'Number of cpu cores to use',
    'num_candidates': 'number of fragmentation trees to compute per feature',
    'tree_timeout': 'time for computation per fragmentation tree in seconds',
    'fingerid_db': 'search structure in given database',
    'maxmz': 'consider compounds with a precursor mz lower or equal to this',
    'zodiac_threshold': 'threshold filter for molecular formula re-ranking',
}

# method registration
plugin.methods.register_function(
    function=fingerprint,
    name='Fingerprint MS data',
    description='Create a contingency table of molecular substructures for a '
                'set of mass spectrometry features',
    inputs={'features': MassSpectrometryFeatures},
    parameters=PARAMS,
    input_descriptions={'features': 'List of MS1 ions and corresponding '
                                    'MS2 ions for each MS1.'},
    parameter_descriptions=PARAMS_DESC,
    outputs=[('collated_fingerprints', FeatureTable[Frequency])],
    output_descriptions={'collated_fingerprints': 'Contingency table of the '
                                                  'probabilities of '
                                                  'molecular substructures '
                                                  'within each feature'}
)

plugin.methods.register_function(
    function=make_hierarchy,
    name='Create a molecular tree',
    description='Build a phylogeny based on molecular substructures',
    inputs={'collated_fingerprints': FeatureTable[Frequency]},
    parameters={'prob_threshold': Float % Range(0, 1, inclusive_end=True)},
    input_descriptions={'collated_fingerprints': 'Contingency table of the '
                                                 'probabilities of '
                                                 'molecular substructures '
                                                 'within each feature'},
    parameter_descriptions={'prob_threshold': 'Probability threshold below '
                                              'which a substructure is '
                                              'considered absent.'},
    outputs=[('tree', Phylogeny[Rooted])],
    output_descriptions={'tree': 'Tree of relatedness between mass '
                                 'spectrometry features based on the chemical '
                                 'substructures within those features'}
)

plugin.methods.register_function(
    function=match_table,
    name='Match feature table to tree tips',
    description='Filters feature table to to match tree tips',
    inputs={'tree': Phylogeny[Rooted],
            'feature_table': FeatureTable[Frequency]},
    input_descriptions={'tree': 'Phylogenetic tree with the features that will'
                                ' be retained on the feature table',
                        'feature_table': 'Feature table that will be filtered '
                                         'based on the features of the '
                                         'phylogenetic tree'},
    parameters={},
    outputs=[('filtered_feature_table', FeatureTable[Frequency])],
    output_descriptions={'filtered_feature_table': 'filtered feature table '
                                                   'that contains only the '
                                                   'features present in '
                                                   'the tree'}
)

plugin.methods.register_function(
    function=collate_fingerprint,
    name='Collate fingerprints into a table',
    description='Collate fingerprints predicted by CSI:FingerID',
    inputs={'csi_result': CSIFingerprintFolder},
    input_descriptions={'csi_result': 'CSI:FingerID output folder'},
    parameters={},
    parameter_descriptions={},
    outputs=[('collated_fingerprints', FeatureTable[Frequency])],
    output_descriptions={'collated_fingerprints': 'Contingency table of the '
                                                  'probabilities of '
                                                  'molecular substructures '
                                                  'within each feature'}
)
