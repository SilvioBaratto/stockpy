import setuptools 

long_description="""
## Description
**stockpy** is a Python Machine Learning library to detect relevant trading patterns and make investmen predictions. At the moment it supports the following algorithms:

- Long Short Term Memory (LSTM)
- Bidirectional Long Short Term Memory (BiLSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional Gated Recurrent Unit (BiGRU)
- Multilayer Perceptron (MLP)
- Gaussian Hidden Markov Models (GHMM)
- Bayesian Neural Networks (BNN)
- Deep Markov Model (DMM)
"""

setuptools.setup(
    name='stockpy-learn',
    version='0.1.8',
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
        "torch>=1.9.0",
        "pyro-ppl>=1.7.0",
        "numpy>=1.20.0",
    ],
)
