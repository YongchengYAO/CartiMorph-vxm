#!/usr/bin/env python

import re
import pathlib
import setuptools

setuptools.dist.Distribution().fetch_build_eggs(['packaging'])
import packaging.version


# base source directory
base_dir = pathlib.Path(__file__).parent.resolve()

# extract the current version
init_file = base_dir.joinpath('CartiMorph_vxm/__init__.py')
init_text = open(init_file, 'rt').read()
pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"
match = re.search(pattern, init_text, re.M)
if not match:
    raise RuntimeError(f'Unable to find __version__ in {init_file}.')
version = match.group(1)

# run setup
setuptools.setup(
    name='CartiMorph_vxm',
    version=version,
    license='Apache 2.0',
    description='Image Registration with Convolutional Networks tailored for the CartiMorph framework',
    url='https://github.com/YongchengYAO/CartiMorph-vxm',
    keywords=['deformation', 'registration', 'imaging', 'cnn', 'mri'],
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'packaging',
        'scikit-image',
        'h5py',
        'numpy',
        'scipy',
        'nibabel',
        'neurite>=0.2',
    ]
)
