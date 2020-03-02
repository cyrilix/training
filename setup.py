import os
from os.path import basename
from os.path import splitext

from glob import glob
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='robocars_sagemaker_container',
    version='1.0.0',

    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={
        'console_scripts': [
            'train=tf_container.train_entry_point:train',
        ]
    },
    install_requires=['sagemaker-container-support'],
    extras_require={},
)
