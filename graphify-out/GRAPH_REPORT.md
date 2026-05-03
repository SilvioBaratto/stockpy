# Graph Report - stockpy  (2026-05-03)

## Corpus Check
- 53 files · ~84,197 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1395 nodes · 2017 edges · 45 communities detected
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 428 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]

## God Nodes (most connected - your core abstractions)
1. `BaseEstimator` - 88 edges
2. `EncoderDecoderForecaster` - 47 edges
3. `DMMForecaster` - 37 edges
4. `TimeSeriesDataset` - 33 edges
5. `TCNForecaster` - 32 edges
6. `BiGRUForecaster` - 31 edges
7. `LSTMForecaster` - 30 edges
8. `GRUForecaster` - 30 edges
9. `TestTimeSeriesDataset` - 29 edges
10. `BiLSTMForecaster` - 28 edges

## Surprising Connections (you probably didn't know these)
- `StockPy Documentation Index` --semantically_similar_to--> `stockpy README`  [INFERRED] [semantically similar]
  docs/index.rst → README.md
- `synthetic_series()` --calls--> `make_synthetic_series()`  [INFERRED]
  test/conftest.py → stockpy/preprocessing/_synthetic.py
- `ts_dataset()` --calls--> `TimeSeriesDataset`  [INFERRED]
  test/conftest.py → stockpy/preprocessing/_dataset.py
- `Design principle: no classification` --rationale_for--> `stockpy.neural_network module`  [INFERRED]
  CLAUDE.md → stockpy/neural_network/__init__.py
- `LSTMRegressor` --references--> `stockpy.neural_network module`  [INFERRED]
  docs/user/gettingstarted.rst → stockpy/neural_network/__init__.py

## Hyperedges (group relationships)
- **RNN-family forecaster test classes sharing encoder-decoder + teacher-forcing + safetensors persistence pattern** — test_lstm_forecaster_testlstmforecaster, test_gru_forecaster_testgruforecaster, test_bilstm_forecaster_testbilstmforecaster, test_bigru_forecaster_testbigruforecaster, test_tcn_forecaster_testtcnforecaster, encoder_decoder_forecaster_contract [INFERRED 0.85]
- **Callback test classes participating in shared History.new_epoch/record protocol with mock_forecaster** — test_callbacks_testearlystopping, test_callbacks_testcheckpoint, test_callbacks_testlrscheduler, test_callbacks_testprintlog, conftest_mock_forecaster_fixture, history_record_pattern [EXTRACTED 1.00]
- **All forecaster test suites verifying EncoderDecoderForecaster subclass contract** — test_forecasters_testforecasterstubs, test_lstm_forecaster_testlstmforecaster, test_dmm_forecaster_testdmmforecaster, encoder_decoder_forecaster_contract [INFERRED 0.95]
- **Callback hook protocol implementers** — callbacks_callback, callbacks_lrscheduler, callbacks_epochtimer, callbacks_printlog, callbacks_scoringbase, callbacks_gradientnormclipping, callbacks_checkpoint, callbacks_earlystopping [EXTRACTED 1.00]
- **DMM encoder-decoder sub-modules** — probabilistic_dmm, probabilistic_combiner, probabilistic_emitterregressor, probabilistic_transition [EXTRACTED 1.00]
- **BaseEstimator collaborators (history, callbacks, utils)** — base_baseestimator, history_history, callbacks_callback, utils_first_step_accumulator, utils_to_tensor, utils_to_device [INFERRED 0.85]
- **All Forecasters implement EncoderDecoderForecaster ABC** — lstm_lstmforecaster, bilstm_bilstmforecaster, gru_gruforecaster, bigru_bigruforecaster, tcn_tcnforecaster, dmm_dmmforecaster, transformer_transformerforecaster, base_encoderdecoderforecaster [EXTRACTED 1.00]
- **RNN-style seq2seq forecasters with teacher forcing** — lstm_lstmforecaster, bilstm_bilstmforecaster, gru_gruforecaster, bigru_bigruforecaster [INFERRED 0.95]
- **Dataset wrappers for numpy-to-PyTorch time-series** — base_stockpydataset, dataset_timeseriesdataset, synthetic_make_synthetic_series, transforms_standardscalertransform [INFERRED 0.85]
- **Stockpy API documentation pages share theme** — neural_network_rst, probabilistic_rst, preprocessing_rst, callbacks_rst, utils_rst [EXTRACTED 1.00]
- **v0.4.0 restructure design principles** — claudemd_no_classification_principle, claudemd_no_flat_regression_principle, claudemd_encoder_decoder_only_principle, claudemd_no_sklearn_dep_principle [EXTRACTED 1.00]
- **Supported algorithms in stockpy README** — readme_bnn_concept, readme_lstm_concept, readme_dmm_concept, readme_nnhmm_concept, readme_mlp_concept [EXTRACTED 1.00]
- **User's Guide topic group** — installation_doc_source, quickstart_doc, core_functions_doc_source [EXTRACTED 1.00]
- **Callback hooks lifecycle** — concept_callback, concept_earlystopping, concept_checkpoint [INFERRED 0.85]
- **stockpy modeling stack** — concept_lstm, concept_bnn, concept_dmm_doc [EXTRACTED 1.00]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.02
Nodes (97): Unpack data returned by the net's iterator into a 2-tuple.      This function is, unpack_data(), BaseEstimator, _current_init_context(), _extract_optimizer_param_name_and_group(), history(), optimizer_setter(), predict() (+89 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (69): EncoderDecoderForecaster, BiGRUModel, Forward pass.          Parameters         ----------         x : torch.Tensor, s, BiGRU-based encoder-decoder for time-series forecasting.      The encoder is a b, Create the BiGRU encoder-decoder module and loss criterion., Reshape encoder hidden for the unidirectional decoder.          PyTorch bidirect, GRUModel, GRU-based encoder-decoder for time-series forecasting.      Parameters     ----- (+61 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (39): DMMForecaster, DMMModel, Generative model over context + future.          Parameters         ----------, Variational guide (inference network).          Parameters         ----------, Deep Markov Model for encoder-decoder time-series forecasting.      The inferenc, Autoregressive prediction from context.          Parameters         ----------, Deep Markov Model encoder-decoder forecaster.      Parameters     ----------, Create the DMM encoder-decoder module. (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (78): Prepare the scoring callback at the beginning of training.          This method, Enum, NotFittedError, Initializes the `StockpyDataset`., Determines if the input is a regular data type for processing.          Regular, Merge input data `x` and additional fitting parameters `fit_params`.          Th, Retrieve parameters for a specific prefix from the instance's dictionary., Retrieve and return initialization parameters for a specific attribute. (+70 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (37): Callback, Basic callback definition., GradientNormClipping, Post-process regularization steps such as gradient normalizing., Clips gradient norm of a module's parameters.      The norm is computed over all, Checkpoint, EarlyStopping, Callbacks related to training progress. (+29 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (20): CausalConv1d, Causal 1-D convolution that maintains temporal ordering.      Left-pads the inpu, TCN encoder + GRU decoder for time-series forecasting.      The TCN encoder proc, Project the last encoder timestep to decoder initial hidden state.          enco, Forward pass.          Parameters         ----------         x : torch.Tensor, s, TCN encoder-decoder forecaster with causal dilated convolutions.      Parameters, Create the TCN encoder-decoder module and loss criterion., Override to set training mode on the PyTorch module. (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.05
Nodes (35): BatchScoring, _cache_net_forward_iter(), cache_net_infer(), convert_sklearn_metric_function(), EpochScoring, PassthroughScoring, Callbacks for calculating scores., Initialize the best score tracking.          Sets the best score to positive or (+27 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (19): DifferenceTransform, Reverse standardization.          Parameters         ----------         data : a, Fit to data, then transform it.          Parameters         ----------         d, First-order (or n-th order) differencing for time series.      Computes discrete, Ensure data is a 2D numpy array., No-op for DifferenceTransform (stateless).          Parameters         ---------, Apply n-th order differencing along the time axis.          Parameters         -, Inverse transform is not defined for differencing.          Raises         ----- (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (53): stockpy.callbacks API, Callbacks User Guide, Callbacks User Guide (source), BaseEstimator, BatchScoring, Bayesian Neural Network (BNN), Callback Base Class, Checkpoint (+45 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (12): GRUForecaster, GRU-based encoder-decoder forecaster.      Parameters     ----------     rnn_siz, Override to set training mode on the PyTorch module., Forward pass through the GRU model.          Parameters         ----------, Override to pass targets to ``infer`` for teacher forcing., Return the state dict of the underlying PyTorch module., Load a state dict into the underlying PyTorch module., make_synthetic_series() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (36): Calculate the weighted average of scores for the latest epoch.          This met, list, Retrieve and return initialization parameters for a specific optimizer., Retrieve initialization parameters for a specified optimizer.          This publ, from_file(), _get_getitem_method(), _getitem_dict_list(), _getitem_dict_str() (+28 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (14): BiLSTMForecaster, BiLSTMModel, Forward pass.          Parameters         ----------         x : torch.Tensor, s, BiLSTM-based encoder-decoder for time-series forecasting.      The encoder is a, BiLSTM-based encoder-decoder forecaster.      Parameters     ----------     rnn_, Create the BiLSTM encoder-decoder module and loss criterion., Override to set training mode on the PyTorch module., Forward pass through the BiLSTM model.          Parameters         ---------- (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (18): _check_lr(), LRScheduler, Simulate the learning rate schedule over a specified number of steps.          T, Initializes the learning rate scheduler.          This method prepares the learn, Ensure a learning rate is provided for each parameter group in the optimizer., Retrieves the learning rate policy class.          This method determines the cl, Initialize the learning rate scheduler at the beginning of training.          Th, Step the learning rate scheduler.          This helper method advances the learn (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.07
Nodes (19): EpochTimer, filter_log_keys(), PrintLog, Callbacks for printing, logging and log information., Initializes the callback before training begins.          This method sets up th, Yield keys from an iterable that are not designated to be ignored for logging., Formats a single row entry for the log output based on the type of value., Sort the keys for displaying in the log output with a predefined order. (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (23): Dataset, _apply_to_data(), get_len(), _is_sparse(), _len(), Get the length of the input data.      If the input data `x` is a sparse matrix,, Get the consistent length of the input data.      This function is particularly, Custom dataset class for Stockpy designed to handle a variety of data types. (+15 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (36): EncoderDecoderForecaster ABC, StockpyDataset, BiGRU nn.Module (legacy), BiGRUForecaster, BiGRUModel encoder-decoder, BiGRURegressor (deprecated, legacy), BiLSTM nn.Module (legacy), BiLSTMForecaster (+28 more)

### Community 16 - "Community 16"
Cohesion: 0.1
Nodes (6): Retrieve the i-th sliding window from the dataset.          Parameters         -, A dataset class for time-series forecasting with context and prediction windows., TimeSeriesDataset, StockpyDataset, Tests for TimeSeriesDataset with context_len and pred_len windows., TestTimeSeriesDataset

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (9): BiGRUForecaster, BiGRU-based encoder-decoder forecaster.      Parameters     ----------     rnn_s, Override to set training mode on the PyTorch module., Forward pass through the BiGRU model.          Parameters         ----------, Override to pass targets to ``infer`` for teacher forcing., Return the state dict of the underlying PyTorch module., Load a state dict into the underlying PyTorch module., Comprehensive tests for BiGRUForecaster encoder-decoder implementation. (+1 more)

### Community 18 - "Community 18"
Cohesion: 0.06
Nodes (35): stockpy.callbacks module, stockpy.callbacks autodoc page, Coding conventions (Black 88, NumPy docstrings, sklearn param pattern), DMM as primary encoder-decoder candidate, Design principle: encoder-decoder only, Design principle: no classification, Design principle: no flat regression, Design principle: no sklearn ClassifierMixin/RegressorMixin (+27 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (33): BaseEstimator (training engine), BatchScoring callback, cache_net_infer context, Callback base class, Checkpoint callback, convert_sklearn_metric_function, EarlyStopping callback, EpochScoring callback (+25 more)

### Community 20 - "Community 20"
Cohesion: 0.14
Nodes (24): Bidirectional encoder + unidirectional decoder with 2*rnn_size hidden, mock_forecaster fixture, MockForecaster fixture class, synthetic_series fixture, ts_dataset fixture, EarlyStopping signals via KeyboardInterrupt, Encoder-Decoder Forecaster contract (context_len + pred_len -> (n,pred_len,n_features)), History new_epoch/new_batch/record protocol (+16 more)

### Community 21 - "Community 21"
Cohesion: 0.1
Nodes (10): Callback, (Re-)Set the initial state of the callback. Use this         e.g. if the callbac, Called at the beginning of training., Called at the end of training., Called at the beginning of each epoch., Called at the end of each epoch., Called at the beginning of each batch., Called at the end of each batch. (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (9): GRU, GRURegressor, Gated Recurrent Unit (GRU) based Recurrent Neural Network for sequence processin, A forecaster that uses a Gated Recurrent Unit (GRU) network for sequence forecas, Initializes the `GRURegressor` instance with the specified configurations., Forward pass through the GRURegressor model.          The method processes the i, Forecast future values for the given input sequences.          Parameters, Constructor for the GRU class.          Initializes a new instance of GRU with t (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (4): mock_forecaster(), MockForecaster, synthetic_series(), ts_dataset()

### Community 24 - "Community 24"
Cohesion: 0.36
Nodes (3): Exception, DataDownloader, main()

### Community 25 - "Community 25"
Cohesion: 0.24
Nodes (10): DataDownloader, DataDownloader.__delete, DataDownloader.__download, DataDownloader.download_stock, main() CLI entrypoint, DataDownloader.__main, DataDownloader.update, DataDownloader.__update_stock (+2 more)

### Community 26 - "Community 26"
Cohesion: 0.4
Nodes (3): _linkcode_resolve(), project_linkcode_resolve(), Determine a link to online source for a class/method/function      This is calle

### Community 27 - "Community 27"
Cohesion: 0.33
Nodes (6): Intersphinx mapping (PyTorch, sklearn, numpy), linkcode_resolve (GitHub source linker), Sphinx Documentation Config, Docs Build Workflow, Custom Sphinx layout template, Sphinx Source Config (Pyro-derived)

### Community 28 - "Community 28"
Cohesion: 0.5
Nodes (5): stockpy solo icon logo (PNG, no wordmark), stockpy solo icon logo (PNG, no wordmark), stockpy logo (PNG, with wordmark), stockpy logo (PNG, with wordmark), stockpy logo (PNG, white variant for dark backgrounds)

### Community 29 - "Community 29"
Cohesion: 0.67
Nodes (4): stockpy solo icon logo (SVG, no wordmark), stockpy solo icon logo (SVG, no wordmark), stockpy logo (SVG, with wordmark), stockpy logo (SVG, with wordmark)

### Community 30 - "Community 30"
Cohesion: 0.67
Nodes (3): _extract_optimizer_param_name_and_group, optimizer_setter helper, _set_optimizer_param

### Community 31 - "Community 31"
Cohesion: 0.67
Nodes (3): check_indexing dispatcher, data_from_dataset, multi_indexing

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (2): TestDifferenceTransform, TestStandardScalerTransform

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (2): stockpy.base module, stockpy package init

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (2): DifferenceTransform, StandardScalerTransform

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Default callbacks used during training and validation processes.          This p

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): A context manager to temporarily set the current initialization context.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Perform a validation step, compute and return the loss, and possibly predictions

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Forecast future time steps for the given input sequences.          Subclasses mu

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Create a History instance from a JSON file.          Parameters         --------

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Retrieves keyword arguments for the scheduler.          This property filters ou

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Constructs the full path for the training history file.          This property a

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): setup() entrypoint

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): get_activation_function

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): ValidSplit

## Knowledge Gaps
- **507 isolated node(s):** `Tests that forecaster stubs can be imported and have the expected interface.`, `Comprehensive tests for LSTMForecaster encoder-decoder implementation.`, `Comprehensive tests for GRUForecaster encoder-decoder implementation.`, `Comprehensive tests for DMMForecaster probabilistic encoder-decoder.`, `Tests for TimeSeriesDataset with context_len and pred_len windows.` (+502 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 33`** (2 nodes): `TestDifferenceTransform`, `TestStandardScalerTransform`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (2 nodes): `stockpy.base module`, `stockpy package init`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (2 nodes): `DifferenceTransform`, `StandardScalerTransform`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Default callbacks used during training and validation processes.          This p`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `A context manager to temporarily set the current initialization context.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Perform a validation step, compute and return the loss, and possibly predictions`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Forecast future time steps for the given input sequences.          Subclasses mu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Create a History instance from a JSON file.          Parameters         --------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Retrieves keyword arguments for the scheduler.          This property filters ou`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Constructs the full path for the training history file.          This property a`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `setup() entrypoint`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `get_activation_function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `ValidSplit`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EncoderDecoderForecaster` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`, `Community 9`, `Community 10`, `Community 11`, `Community 17`, `Community 22`, `Community 23`?**
  _High betweenness centrality (0.303) - this node is a cross-community bridge._
- **Why does `BaseEstimator` connect `Community 0` to `Community 1`, `Community 10`, `Community 3`, `Community 21`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Why does `History` connect `Community 10` to `Community 0`, `Community 1`, `Community 4`, `Community 12`, `Community 13`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `BaseEstimator` (e.g. with `EncoderDecoderForecaster` and `BaseEstimator`) actually correct?**
  _`BaseEstimator` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 42 inferred relationships involving `EncoderDecoderForecaster` (e.g. with `BaseEstimator` and `TestForecasterStubs`) actually correct?**
  _`EncoderDecoderForecaster` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `DMMForecaster` (e.g. with `EncoderDecoderForecaster` and `Combiner`) actually correct?**
  _`DMMForecaster` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `TimeSeriesDataset` (e.g. with `.test_constructor_accepts_context_len_pred_len_stride()` and `.test_getitem_returns_tuple_of_tensors()`) actually correct?**
  _`TimeSeriesDataset` has 28 INFERRED edges - model-reasoned connections that need verification._