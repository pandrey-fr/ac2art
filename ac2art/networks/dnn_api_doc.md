### The NeuralNetwork API

#### Basic user-oriented API

This section presents the public methods inherited from the
`NeuralNetwork` class that should be used to build, train,
use and save models.

**Initialization: layers configuration and data shape (Basic API, 1/3)**

Parameters of the instanciation method inherited from NeuralNetwork
(then enhanced to suit the specificities of the various models) make
the networks' input shape, stack of hidden layers and output shape
fully modular.

- The `input_shape` parameter is two-fold: its sets up the input
  vectors' dimension, but also allows to make the network suitable
  for batch learning. To compute batches of 1-D input vectors, set
  up the argument to `(None, input_length)`. Note that his also allows
  to process single time series of input vectors. To process such time
  series by batches of free size, concatenating series of variable
  length, set up the argument to `(None, None, input_length)`.

    _Note that batching the inputs is done by the model: passing
  a sequence (list, array, tuple...) of 2-D arrays is sufficient._

- The `layers_config` argument fully specifies the network's hidden
  layers stack. The structure of `layers_config` is rather straight-forward:
  it consists of a list of tuples, each of which specifies a layer of
  the network, ordered from input to readout and stacked on top of
  each other. These layers may either be an actual neural layer, a
  stack of RNN layers or a signal filtering process. Each layer is
  specified as a tuple containing the layer's class (or a keyword
  designating it), its number of units (or cutoff frequency, for
  signal filters) and an optional dict of keyword arguments used
  to instantiate the layer.

- The `top_filter` argument allows to specify an additional signal
  filter on top of the readout layer. This is useful when processing
  time sequences of input vectors, and thus relevant for learning
  acoustic-to-articulatory inversion. This argument should be a tuple
  similar to those listed in `layers_config`.

- The `n_targets` argument is an integer equal to the length
  of the (time-wise) output vectors. Additional modularity is
  provided by the `use_dynamic` argument, which adds delta and
  deltadelta features to these dimensions, and by the `binary_tracks`
  one, which allows to designate given dimensions of the output
  as binary values instead of continuous ones: these values will
  have their own readout layer parallel to that of the others on
  top of the shared stack of hidden layers, their own loss function
  (minimized jointly with that of the continuous targets) and will
  of course not be taken into account when computing delta features.


**Training, predicting and scoring methods (Basic API, 2/3)**

- The `run_training_function` should be used to train the model.
  To be more precise, it triggers a single training epoch, based
  on both some input data and the associated targets to run. For
  some models, an additional `loss` parameter allows to choose
  between multiple loss functions to backpropagate on. For all
  models, the `keep_prob` argument may be set to any float
  between 0 and 1 to use dropout when training the layers.
  Note that by default, all layers are set to be affected
  by dropout, with a shared probability parameter ; this may
  be changed by explicitly setting 'keep_prob' to None in these
  layers' keyword arguments dict in `layers_config` at instanciation.

- The `predict` method requires only some input data and returns
  the network's prediction as a numpy.ndarray. For batched inputs,
  a flat array of arrays is returned, with each of the latter
  recording the prediction associated with an input vector (in the
  same order as provided).

- The `score` method returns a subclass-specific evaluation metric
  of the model's outputs based on some input data and the target
  values associated with it. For models designed to process batches
  of input vectors, using this method on such a batch will have the
  model return sequence-wise metrics.

- The `score_corpus` method is similar to the `score` one, but is
  applied to sequences of input vectors (which may not be batched)
  and synthetised on the overall. Note that one can pass an iterable
  that reads the data on the go to this function, e.g. to compute
  synthetic metrics on a corpus too large to fit in memory.


**Saving, restoring and resetting the model (Basic API, 3/3)**

- The `save_model` method allows to save the network's weights as
  well as its full specification to a simple .npy file. The stored
  values may also be accessed through the `architecture` attribute
  and the `get_values` method.

- The `restore_model` method allows to restore and instantiated
  model's weights from a .npy dump. More generally, the function
  `load_dumped_model` may be used to fully instantiate a dumped
  model.

- The `reset_model` method may be used at any moment to reset the
  model's weights to a randomized value, as if newly initialized.



#### NeuralNetwork API for developers

Apart from designing some common arguments, the `__init__` method
includes both an arguments-processing procedure which enables its
call by any subclass (bypassing intermediary subclasses if needed)
and a network building procedure. The latter is made of multiple
private methods called successively and protected against being
called more than once. This section aims at presenting the design
of these hidden methods, some of which are meant to be overridden
by subclasses.

**Setting up the network's basics and hidden stack (Network building, 1/2)**

- The `_validate_args` method is first run to ensure that all
  arguments provided to instantiate a network are of expected
  type and/or values. Subclasses should override this method
  to validate any non-basic `__init__` parameter they introduce.

- The `_build_placeholders` method is then run to assign tensorflow
  placeholders to the dict attribute `holders`. Those are used to
  pass on input and target data, but also to specify parameters
  such as dropout. Subclasses may have to override this, either
  to introduce additional placeholders or alter the shape of the
  basic ones.

- The `_build_hidden_layers` method comes next, and is run to
  instantiate forward layers, recurrent units' stacks and signal
  filters following the architecture specified through the
  `layers_config` attribute. This method should not need overriding
  by any subclass, as it is a general way to build up hidden layers
  sequentially, handling some technicalities such as setting dropout
  (unless explicitly told not to) or assigning unique names to
  rnn stacks in order to avoid tensorflow scope issues. The hidden
  layers are stored in the `layers` OrderedDict attribute.


**From readouts to training - abstract methods (Network building, 2/2)**

- The `_build_readout_layer` is an abstract method that needs
  implementing by subclasses. It should design the network's final
  hidden layer (typically using an identity activation function),
  whose purpose is to produce an output of proper dimension to be
  then used to derive a prediction, or any kind of metric useful
  to train the network.

- The `_build_readouts` method is run after the previous, and is
  basically a caller of other hidden methods used sequentially
  to fill the `readouts` dict attribute with tensors useful to
  train and/or evaluate the network's performances. This method
  may be overridden by subclasses which may need to add up steps
  (i.e. additional methods) to this end. In its basic definition,
  this method calls, in that order, the following methods:
    1. `build_initial_prediction`, an abstract method which should
    assign a tensor to the `readouts` attribute under the
    'raw_prediction' key.

    2. `build_refined_prediction`, an implemented method which aims
    at improving the raw prediction, through optional steps of
    de-normalization and signal filtering (smoothing).

    3. `build_error_readouts`, an abstract method which should assign
    to the `readouts` attribute any tensor necessary to building
    the training function.

- Finally, the `_build_training_function` is run. This abstract
  method should build one or more tensorflow operations that
  need running so as to update the network's weights (or signal
  cutoff frequencies) and assign them (e.g. as a list) to the
  `training_function` attribute.
