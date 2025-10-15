from setuptools import setup, find_packages

# Setup script for the pediatric_appendicitis_labelling package.
# This file is used by packaging tools like pip to install the project and its dependencies.

setup(
    # The name of the package
    name='pediatric_appendicitis_labelling',

    # The version of the package
    version='0.1.0',

    # A short description of the package
    description='A project to label pediatric appendicitis cases from clinical notes.',

    # The author of the package
    author='Abhishek Paul',

    # The author's email address
    author_email='abhk.paul@gmail.com',

    # Automatically find all packages in the project (i.e., directories with an __init__.py file)
    packages=find_packages(),

    # List of dependencies required for this package to run
    # These will be installed by pip when the package is installed.
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'pyyaml',
        'pdfplumber',
        'pypdf2',
        'xgboost',
        'lightgbm',
        'gitpython',
        'spacy',
        'openai',
        'matplotlib',
        'seaborn',
        'tqdm'
    ],

    # Provides metadata about the package for package indexes
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    # Specifies the minimum required version of Python
    python_requires='>=3.8',
)