from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "This package is designed to work with Python 3.6 and greater."

setup(
    name='parviflora',
    py_modules=['parviflora'],
    version='0.1',
    install_requires=[
        'gymnasium',
        # 'joblib',
        # 'matplotlib==3.1.1',
        'numpy',
        # 'pandas',
        # 'seaborn',
        'torch',
        'tqdm',
        'tensorboard'
    ],
    description="RL",
    author="Paul Gajewski",
)
