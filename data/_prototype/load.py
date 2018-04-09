# coding: utf-8

"""Set of functions to load pre-processed mngu0 data."""

import os

import numpy as np

from data.commons.enhance import build_context_windows
from data._prototype.normalize import _get_normfile_path
from data.utils import CONSTANTS, import_from_string


def build_setup_functions(corpus, default_byspeaker):
    """Build functions to view and alter a dict of loading arguments."""
    # Declare the loading setup dict.
    loading_setup = {
        'audio_type': 'mfcc_stds' + ('_byspeaker' if default_byspeaker else ''),
        'context_window': 5,
        'dynamic_ema': True,
        'ema_norm': 'mean' + ('_byspeaker' if default_byspeaker else ''),
        'zero_padding': True
    }
    # Define functions to manipulate the former dict.
    def change_loading_setup(
            audio_type=None, context_window=None, dynamic_ema=None,
            ema_norm=None, zero_padding=None
        ):
        """Update the default arguments used when importing {0} data.

        The parameters set through this function are those used by default
        by functions `load_utterance`, and `load_dataset` functions of the
        `data.{0}.load` module.

        audio_type     : name of the audio features to use,
                         including normalization indications (str)
        context_window : half-size of the context window of acoustic inputs
                         (set to zero to use single audio frames as input)
        dynamic_ema    : whether to use dynamic articulatory features
        ema_norm       : optional type of normalization of the EMA data
                         (set to '' to use raw EMA data)
        zero_padding   : whether to use zero-padding when building context
                         windows, or use edge frames as context only

        Any number of the previous parameters may be passed to this function.
        All arguments left to their default value of `None` will go unchanged.

        To see the current value of the parameters, use `get_loading_setup`.
        """
        # Broadly catch all arguments; pylint: disable=unused-argument
        kwargs = locals()
        nonlocal loading_setup
        for key, argument in kwargs.items():
            if argument is not None and key != 'loading_setup':
                loading_setup[key] = argument

    def get_loading_setup():
        """Consult the current default arguments for importing {0} data.

        Please note that modifying the returned dict does not alter
        the actual setup. To change those, use `change_loading_setup`.
        """
        return loading_setup.copy()

    # Adjust the former two functions' docstrings and return them.
    change_loading_setup.__doc__ = change_loading_setup.__doc__.format(corpus)
    get_loading_setup.__doc__ = get_loading_setup.__doc__.format(corpus)
    return change_loading_setup, get_loading_setup


def build_file_loaders(corpus):
    """Define and return functions to load single-file data or parameters."""
    # Load dependency path and function.
    data_folder = CONSTANTS['%s_processed_folder' % corpus]
    get_utterances_list = import_from_string(
        'data.%s.raw._loaders' % corpus, 'get_utterances_list'
    )
    # Define the four loading functions.
    def get_norm_parameters(file_type, speaker=None):
        """Return normalization parameters for a type of {0} features.

        file_type : type of features whose parameters to return (str)
        speaker   : optional speaker whose parameters to return (str)
                    (otherwise, corpus-wide parameters are returned)
        """
        nonlocal data_folder
        path = _get_normfile_path(data_folder, file_type, speaker)
        return np.load(path).tolist()

    def get_utterances(set_name=None):
        """Get the list of utterances from a given set.

        set_name : name of the set, e.g. 'train', 'validation' or 'test'
        """
        nonlocal data_folder, get_utterances_list
        if set_name is None:
            return get_utterances_list()
        path = os.path.join(data_folder, 'filesets', set_name + '.txt')
        with open(path) as file:
            utterances = [row.strip('\n') for row in file]
        return utterances

    def load_acoustic(
            name, audio_type='mfcc_stds', context_window=0, zero_padding=True
        ):
        """Load the acoustic data associated with an utterance from {0}.

        name           : name of the utterance whose data to load (str)
        audio_type     : name of the audio features to use, including
                         normalization indications (str, default 'mfcc_stds')
        context_window : half-size of the context window of frames to return
                         (default 0, returning single audio frames)
        zero_padding   : whether to zero-pad the data when building context
                         frames (bool, default True)
        """
        nonlocal data_folder
        audio_type, norm_type = (audio_type + '_').split('_', 1)
        folder = (
            audio_type + '_norm_' + norm_type.strip('_')
            if norm_type else audio_type
        )
        path = os.path.join(data_folder, folder, name + '_%s.npy' % audio_type)
        acoustic = np.load(path)
        if context_window:
            return build_context_windows(acoustic, context_window, zero_padding)
        return acoustic

    def load_ema(name, norm_type='', use_dynamic=True):
        """Load the articulatory data associated with an utterance from {0}.

        name           : name of the utterance whose data to load (str)
        norm_type      : optional type of normalization to use (str)
        use_dynamic    : whether to return dynamic features (bool, default True)
        """
        nonlocal data_folder, get_norm_parameters
        ema_folder = (
            'ema' if norm_type in ('', 'mean', 'mean_byspeaker')
            else 'ema_norm_' + norm_type
        )
        ema = np.load(os.path.join(data_folder, ema_folder, name + '_ema.npy'))
        if norm_type.startswith('mean'):
            speaker = None if norm_type == 'mean' else name.split('_', 1)[0]
            ema -= get_norm_parameters('ema', speaker)['global_means']
        if use_dynamic:
            return ema
        return ema[:, :ema.shape[1] // 3]

    # Adjust the functions' docstrings and return them.
    functions = (get_norm_parameters, get_utterances, load_acoustic, load_ema)
    for function in functions:
        function.__doc__ = function.__doc__.format(corpus)
    return functions


def build_loading_functions(corpus, default_byspeaker):
    """Define and return data loading functions for a given corpus."""
    # Use auxiliary functions to build the basic loading functions.
    change_loading_setup, get_loading_setup = (
        build_setup_functions(corpus, default_byspeaker)
    )
    get_norm_parameters, get_utterances, load_acoustic, load_ema = (
        build_file_loaders(corpus)
    )
    # Define the major loading functions.
    def load_utterance(name, **kwargs):
        """Load both acoustic and articulatory data of an mngu0 utterance.

        name : name of the utterance whose data to load (str)

        Any keyword argument of functions `load_acoustic` and `load_ema`
        may be passed ; otherwise, current setup values will be used.
        The latter can be seen and changed using `get_loading_setup`
        and `change_loading_setup`, all from the `data.mngu0.load` module.
        """
        nonlocal get_loading_setup, load_acoustic, load_ema
        args = get_loading_setup()
        args.update(kwargs)
        acoustic = load_acoustic(
            name, args['audio_type'], args['context_window'],
            args['zero_padding']
        )
        ema = load_ema(name, args['ema_norm'], args['dynamic_ema'])
        if args['context_window'] and not args['zero_padding']:
            ema = ema[args['context_window']:-args['context_window']]
        return acoustic, ema

    def load_dataset(set_name, concatenate=False, **kwargs):
        """Load the acoustic and articulatory data of an entire {0} fileset.

        set_name    : name of the fileset, e.g. 'train', 'validation' or 'test'
        concatenate : whether to concatenate the utterances into a single
                      numpy.ndarray (bool, default False)

        Any keyword argument of functions `load_acoustic` and `load_ema`
        may be passed ; otherwise, current setup values will be used.
        The latter can be seen and changed using `get_loading_setup`
        and `change_loading_setup`, all from the `data.{0}.load` module.
        """
        nonlocal get_utterances, load_utterance
        fileset = get_utterances(set_name)
        dataset = np.array([load_utterance(name, **kwargs) for name in fileset])
        acoustic = np.array([utterance[0] for utterance in dataset])
        ema = np.array([utterance[1] for utterance in dataset])
        if concatenate:
            return np.concatenate(acoustic), np.concatenate(ema)
        return acoustic, ema

    # Adjust the latter two functions' docstrings.
    load_utterance.__doc__ = load_utterance.__doc__.format(corpus)
    load_dataset.__doc__ = load_dataset.__doc__.format(corpus)
    # Return the all set of loading functions.
    return (
        change_loading_setup, get_loading_setup, get_norm_parameters,
        get_utterances, load_acoustic, load_ema, load_utterance, load_dataset
    )
