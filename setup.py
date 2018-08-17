# ----------------------------------------------------------------------------
# Copyright (c) 2016-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from setuptools import setup, find_packages
import versioneer

setup(
    name='q2-chemistree',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    author='Anupriya Tripathi',
    author_email='a3tripat@ucsd.edu',
    description='Molecular tree inference for metabolomics analysis.',
    license='BSD-2-Clause',
    url="https://github.com/biocore/chemistree",
    entry_points={
        'qiime2.plugins': ['q2-chemistree=q2_chemistree.plugin_setup:plugin']
    }
)
