Installation
============

.. contents::
   :local:


pip installation
~~~~~~~~~~~~~~~~

To install with pip, run:

.. code:: bash

    python -m pip install -U stockpy-learn

We recommend to use a virtual environment for this.

From source
~~~~~~~~~~~

If you would like to use the most recent additions to stockpy or
help development, you should install stockpy from source.

You may adjust the Python version to any of the supported Python versions, i.e.
Python 3.8 or higher.

Using pip
^^^^^^^^^

If you just want to use skorch, use:

.. code:: bash

    git clone https://github.com/SilvioBaratto/stockpy.git
    cd skorch
    # create and activate a virtual environment
    python -m pip install -r requirements.txt
    # install pytorch version for your system (see below)
    python -m pip install .
