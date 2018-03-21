[Public API]

* Initialization and layers configuration (Public API, 1/3)

- An `__init__` method allows the user to fully configure the
  network's architecture and specificities, noticeably through
  the `layers_config` argument presented below. The `__init__`
  is further discussed in the "Network building" section.

- The `layers_config` argument used at the initialization step
  fully specifies the network's hidden layers stack, and should
  do so in any subclass. Subclasses should in turn specify the
  readout layer, the algorithm generating a prediction out of
  it and the training function(s) used to fit the model.

- The structure of `layers_config` is rather straight-forward: it
  consists of a list of tuples, each of which specifies a layer of
  the network, ordered from input to readout and stacked on top of
  each other. These layers may either be an actual neural layer, a
  stack of RNN layers or a signal filtering process. Each layer is
  specified as a tuple containing the layer's class (or a keyword
  designating it), its number of units (or cutoff frequency, for
  signal filters) and an optional dict of keyword arguments used
  to instanciate the layer.


* Training, predicting and scoring methods (Public API, 2/3)

- The `run_training_function` should be used to train the model.
  It requires both some input data and the associated targets to
  run. Additionally, the `keep_prob` argument may be set to any
  float between 0 and 1 to use dropout when training the layers.
  Note that by default, all dense layers are set to be affected
  by this dropout, with a shared probability parameter ; this may
  be changed by explicitly setting 'keep_prob' to None in these
  layers' keyword arguments dict in `layers_config` at `__init__`.

- The `predict` method requires only input data and returns the
  network's prediction as a numpy.array.

- The `score` method returns a subclass-specific evaluation metric
  of the model's outputs based on some input data and the target
  values associated with it.


* Saving, restoring and resetting the model (Public API, 3/3)

- The `save_model` method allows to save the network's weights as
  well as its full specification to a simple .npy file. The stored
  values may also be accessed through the `architecture` attribute
  and the `get_values` method.

- The `restore_model` method allows to restore and instanciated
  model's weights from a .npy dump. More generally, the function
  `load_dumped_model` may be used to fully instanciate a dumped
  model.

- The `reset_model` method may be used at any moment to reset the
  model's weights to their initial (randomized) value.


[Network building]

Apart from designing some common arguments, the `__init__` method
includes both an arguments-processing procedure which enables its
call by any subclass (bypassing intermediary subclasses if needed)
and a network building procedure. The latter is made of multiple
private methods called successively and protected against being
called more than once. This section aims at presenting the design
of these hidden methods, some of which are meant to be overridden
by subclasses.

* Setting up the network's basics and hidden stack (Network building, 1/2)

- The `_validate_args` method is first run to ensure that all
  arguments provided to instanciate a network are of expected
  type and/or values. Subclasses should override this method
  to validate any non-basic `__init__` parameter they introduce.

- The `_build_placeholders` method is then run to assign tensorflow
  placeholders to the dict attribute `_holders`. Those are used to
  pass on input and target data, but also to specify parameters
  such as dropout. Subclasses may have to override this, either
  to introduce additional placeholders or alter the shape of the
  basic ones.

- The `_build_hidden_layers` method comes next, and is run to
  instanciate forward layers, recurrent units' stacks and signal
  filters following the architecture specified through the
  `layers_config` attribute. This method should not need overriding
  by any subclass, as it is a general way to build up hidden layers
  sequentially, handling some technicalities such as setting dropout
  (unless explicitly told not to) or assigning unique names to
  rnn stacks in order to avoid tensorflow scope issues. The hidden
  layers are stored in the `_layers` OrderedDict attribute.


* From readouts to training - abstract methods (Network building, 2/2)

- The `_build_readout_layer` is an abstract method that needs
  implementing by subclasses. It should design the network's final
  hidden layer (typically using an identity activation function),
  whose purpose is to produce an output of proper dimension to be
  then used to derive a prediction, or any kind of metric useful
  to train the network.

- The `_build_readouts` method is run after the previous, and is
  basically a caller of other hidden methods used sequentially
  to fill the `_readouts` dict attribute with tensors useful to
  train and/or evaluate the network's performances. This method
  may be overridden by subclasses which may need to add up steps
  (i.e. additional methods) to this end. In its basic definition,
  this method calls, in that order, the following methods:
    1. `build_initial_prediction`, an abstract method which should
    assign a tensor to the `_readouts` attribute under the
    'raw_prediction' key.

    2. `build_refined_prediction`, an implemented method which aims
    at improving the raw prediction, through optional steps of
    de-normalization and signal filtering (smoothing).

    3. `build_error_readouts`, an abstract method which should assign
    to the `_readouts` attribute any tensor necessary to building
    the training function.

- Finally, the `_build_training_function` is run. This abstract
  method should build one or more tensorflow operations that
  need running so as to update the network's weights (or signal
  cutoff frequencies) and assign them (e.g. as a list) to the
  `_training_function` attribute.


[Network training and scoring]

- `run_training_function`
- `predict`
- `score`
