Callbacks
=========

Callbacks in the Stockpy library offer a versatile mechanism to insert custom behavior into the training process of models without the necessity of modifying the base classes. This document will outline the core aspects of implementing callbacks and the common methods available for customization.

Overview of Callbacks
---------------------

Callbacks are not exhaustively detailed here. For a comprehensive list and explanations, refer to the Stockpy callback module documentation.

Base Class for Callbacks
------------------------

All callbacks derive from the :class:`.Callback` base class. Custom callbacks should be created by subclassing :class:`.Callback` and overriding its methods to perform the desired actions at each stage of the training process.

Here are the key points to remember when implementing a custom callback:

- Inherit from :class:`.Callback` base class.
- Override one or more :code:`on_` methods that hook into the training lifecycle.
- Callback methods accept the :class:`.BaseEstimator` instance as the first argument, followed by context-specific parameters like batch data or model parameters, with :code:`**kwargs` for flexibility.

Essential Callback Methods
--------------------------

The methods below are the hooks available to customize the behavior of your callback:

- :code:`initialize()`: Set or reset any necessary attributes when the model initializes.
- :code:`on_train_begin(net, X, y)`: Invoked once before the training starts.
- :code:`on_train_end(net, X, y)`: Invoked once after training finishes.
- :code:`on_epoch_begin(net, dataset_train, dataset_valid)`: Invoked at the start of each epoch.
- :code:`on_epoch_end(net, dataset_train, dataset_valid)`: Invoked at the end of each epoch.
- :code:`on_batch_begin(net, batch, training)`: Invoked before each batch is processed.
- :code:`on_batch_end(net, batch, training, loss, y_pred)`: Invoked after each batch is processed.

Implementing Scoring Callbacks
------------------------------

Stockpy provides :class:`.EpochScoring` and :class:`.BatchScoring` for evaluating models at different stages of the training. The scoring callbacks are utilized when additional metrics are required beyond the defaults provided by :class:`.BaseEstimator`. These callbacks are characterized by parameters like ``name``, ``lower_is_better``, ``on_train``, and possibly a ``target_extractor`` for preprocessing targets before scoring.

For instance, :class:`.EpochScoring` is suited for scores requiring epoch-wide data, while ``BatchScoring`` is beneficial for memory-intensive tasks.

Here is an example setup for early stopping using a callback:

.. code:: python

    from skorch.callbacks import EarlyStopping
    
    
    early_stopping = EarlyStopping(
    			monitor='valid_loss',
    			patience=5,
    			threshold=0,
    			threshold_mode='rel',
    			lower_is_better=True)
          
    predictor.fit(X_train, 
                  y_train,
                  batch_size=32,
                  lr=0.01, 
                  optimizer=torch.optim.Adam,
                  callbacks=[early_stopping],
                  epochs=50)


Checkpoint
----------

:class:`.Checkpoint` callback saves model checkpoints based on certain conditions, such as improvement in validation loss. Customize checkpoint behavior with parameters like ``f_params``, ``f_optimizer``, ``f_history``, and ``f_pickle``. Here is how to set up a model checkpointing:

.. code:: python

    from skorch.callbacks import Checkpoint
    
    checkpoint = Checkpoint(
    			f_params='best_model_params.pt',
    	    		monitor='valid_loss_best',             	                
    	    		f_optimizer='best_optimizer_params.pt', 
    	    		f_history='best_model_history.json',  
    	    		)
          
    predictor.fit(X_train, 
                  y_train,
                  batch_size=32,
                  lr=0.01, 
                  optimizer=torch.optim.Adam,
                  callbacks=[checkpoint],
                  epochs=50)


Learning rate schedulers
------------------------

The :class:`.LRScheduler` callback integrates with :mod:`torch.optim.lr_scheduler` for dynamic learning rate adjustments. For example, to implement a step-based learning rate scheduler, use the following setup:

.. code:: python

    from skorch.callbacks import LRScheduler
    from torch.optim.lr_scheduler import StepLR
    
    # Define the LR scheduler callback
    lr_scheduler = LRScheduler(policy=StepLR, step_size=10, gamma=0.7)
          
    predictor.fit(X_train, 
                  y_train,
                  batch_size=32,
                  lr=0.01, 
                  optimizer=torch.optim.Adam,
                  callbacks=[lr_scheduler],
                  epochs=50)
    
    
    
    
    
    
    
    
    
    
    
