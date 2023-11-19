import os

from setuptools import setup, find_packages

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

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
except IOError:
    README = ''
    
setup(
    name='stockpy',
    version=version,
    description='Deep Learning Regression and Classification Library built on top of PyTorch and Pyro',
    long_description=README,
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/SilvioBaratto/stockpy",
    zip_safe=False,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require={
        'docs': docs_require,
    },
)
