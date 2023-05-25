import setuptools 

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='stockpy-learn',
    version='0.2.1',
    author='Silvio Baratto',
    author_email='silvio.baratto22@gmail.com',
    description='Machine Learning library to make stock market prediction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SilvioBaratto/stockpy",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent", 
        'Programming Language :: Python :: 3',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.9.1",
        "pyro-ppl>=1.7.0",
        "numpy>=1.21.2",
        "pandas>=1.3.3",
        "yfinance>=0.1.63",
        "yahoofinancials>=1.6",
        "pandas-datareader>=0.10.0",
        "tqdm>=4.62.3",
        "scikit-learn>=0.24.2",
    ],
)
