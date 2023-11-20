import os

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('VERSION', 'r') as f:
    version = f.read().rstrip()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]

python_requires = '>=3.8'

docs_require = [
    'Sphinx',
    'sphinx_rtd_theme',
    'numpydoc',
]

setup(
    name='stockpy-learn',
    version=version,
    author='Silvio Baratto',
    author_email='silvio.baratto22@gmail.com',
    description='Deep Learning Regression and Classification Library built on top of PyTorch and Pyro',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/SilvioBaratto/stockpy",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent", 
        'Programming Language :: Python :: 3',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
    },
)
