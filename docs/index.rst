StockPy Documentation
=====================

Introduction
------------

``stockpy`` is a Python Machine Learning library originally developed for stock market data analysis and prediction. Now it has expanded its reach to provide advanced tools for a wide array of datasets, offering both regression and classification capabilities.

Powered by PyTorch and Pyro, ``stockpy`` provides users with an assortment of algorithms, each equipped for regression and classification tasks. These include Bayesian Neural Networks (BNN), various configurations of Recurrent Neural Networks like LSTM and GRU, as well as advanced models such as Deep Markov Models and Gaussian Hidden Markov Models.

The library is designed with simplicity and flexibility in mind, making it accessible for both novice and expert data scientists.

.. code:: bibtex

   @manual{stockpy,
     author       = {Silvio Baratto},
     title        = {StockPy: A Versatile Machine Learning Library for Data Analysis and Prediction},
     month        = nov,
     year         = 2023,
     url          = {https://stockpy.readthedocs.io/en/latest/}
   }

User's Guide
------------

The following sections are included in the User's Guide to help you get started with ``stockpy``:

.. toctree::
   :glob:
   :maxdepth: 2

   user/installation
   user/gettingstarted
   user/classification
   user/regression
   user/neuralnetwork
   user/probabilistic
   user/callbacks
   user/preprocessing

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   stockpy API <stockpy>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Getting Help
------------

We welcome contributions, feedback, and bug reports.

.. _PyTorch: https://pytorch.org/
.. _Pyro: https://pyro.ai/

