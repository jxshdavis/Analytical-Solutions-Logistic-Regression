from setuptools import setup, find_packages

setup(
    name='AnalyticSolution',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.2.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.24.0',
        'scipy>=1.5.0',
        'pandas>=1.1.0',
    ],
)

