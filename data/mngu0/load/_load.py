# coding: utf-8

"""Set of functions to load pre-processed mngu0 data."""

import os

import numpy as np

from data.commons import add_dynamic_features, build_context_windows
from data.utils import load_data_paths


_, FOLDER = load_data_paths('mngu0')


_SETUP = {
    'audio_types': ('mfcc_stds', 'energy'),
    'context_window': 5,
    'dynamic_audio': True,
    'dynamic_ema': False,
    'dynamic_window': 5,
    'ema_norm': 'mean',
    'zero_padding': True
}


def change_loading_setup(
        audio_types=None, context_window=None, dynamic_audio=None,
        dynamic_ema=None, dynamic_window=None, ema_norm=None, zero_padding=None
    ):
    """Update the default arguments used when importing mngu0 data.

    The parameters set through this function are those used by default
    by functions `load_utterance`, and `load_dataset` functions of the
    `data.mngu0.load` module.

    audio_types    : tuple of names of audio features to use,
                     including normalization indications
    context_window : half-size of the context window of acoustic inputs
                     (set to zero to use single audio frames as input)
    dynamic_audio  : whether to compute dynamic audio features
    dynamic_ema    : whether to compute dynamic articulatory features
    dynamic_window : half-size of the window used when computing dynamic
                     features
    ema_norm       : optional type of normalization of the EMA data
                     (set to '' to use raw EMA data)
    zero_padding   : whether to use zero-padding when building context
                     windows, or use edge frames as context only

    Any number of the previous parameters may be passed to this function.
    All arguments left to their default value of `None` will go unchanged.

    To see the current value of the parameters, use `see_loading_setup`.
    """
    # Arguments serve modularity; pylint: disable=too-many-arguments
    # Broadly catch all arguments; pylint: disable=unused-argument
    kwargs = locals()
    for key, argument in kwargs.items():
        if argument is not None:
            _SETUP[key] = argument


def see_loading_setup():
    """Consult the current default arguments used when importing mngu0 data.

    Please note that modifying the returned dict does not alter
    the actual setup. To change those, use `change_loading_setup`.
    """
    return _SETUP.copy()


def get_normalization_parameters(file_type):
    """Return normalization parameters for a type of mngu0 features."""
    path = os.path.join(FOLDER, 'norm_params', 'norm_%s.npy' % file_type)
    return np.load(path).tolist()


def load_acoustic(
        name, audio_types=('mfcc_stds',), context_window=0,
        dynamic_window=0, zero_padding=True
    ):
    """Load the acoustic data associated with an utterance from mngu0.

    name           : name of the utterance whose data to load (str)
    audio_types    : tuple of names of audio features to use,
                     including normalization indications
    context_window : half-size of the context window of frames to return
                     (default 0, returning single audio frames)
    dynamic_window : half-size of the window used when computing dynamic
                     features (default 0, returning static features only)
    zero_padding   : whether to zero-pad the data when building context
                     frames (bool, default True)
    """
    acoustic = np.concatenate([
        _load_acoustic(name, audio_type) for audio_type in audio_types
    ], axis=1)
    if dynamic_window:
        acoustic = add_dynamic_features(acoustic, dynamic_window)
    if context_window:
        acoustic = build_context_windows(acoustic, context_window, zero_padding)
    return acoustic


def _load_acoustic(name, audio_type):
    """Load some acoustic data associated with an mngu0 utterance.

    name       : name of the utterance whose data to return
    audio_type : type of features loaded, optionally comprising
                 a type of normalization separated by an underscore
    """
    audio_type, norm_type = (audio_type + '_').split('_', 1)
    folder = (
        audio_type + '_norm_' + norm_type.strip('_') if norm_type
        else audio_type
    )
    return np.load(os.path.join(FOLDER, folder, name + '_%s.npy' % audio_type))


def load_ema(name, norm_type='', dynamic_window=0):
    """Load the articulatory data associated with an utterance from mngu0.

    name           : name of the utterance whose data to load (str)
    norm_type      : optional type of normalization to use (str)
    dynamic_window : half-size of the window used when computing dynamic
                     articulatory features (default 0, returning static
                     features only)
    """
    ema_dir = 'ema' if norm_type in ('', 'mean') else 'ema_norm_' + norm_type
    ema = np.load(os.path.join(FOLDER, ema_dir, name + '_ema.npy'))
    if norm_type == 'mean':
        ema -= get_normalization_parameters('ema')['global_means']
    if dynamic_window:
        ema = add_dynamic_features(ema, window=dynamic_window)
    return ema


def load_utterance(name, **kwargs):
    """Load both acoustic and articulatory data of an mngu0 utterance.

    name : name of the utterance whose data to load (str)

    Any keyword argument of functions `load_acoustic` and `load_ema`
    may be passed ; otherwise, current setup values will be used.
    The latter can be seen and changed using `see_loading_setup`
    and `change_loading_setup`, all from the `data.mngu0.load` module.
    """
    args = _SETUP.copy()
    args.update(kwargs)
    acoustic = load_acoustic(
        name, args['audio_types'], args['context_window'],
        dynamic_window=args['dynamic_window'] if args['dynamic_audio'] else 0,
        zero_padding=args['zero_padding']
    )
    ema = load_ema(
        name, args['ema_norm'],
        dynamic_window=args['dynamic_window'] if args['dynamic_ema'] else 0
    )
    if not args['zero_padding'] and args['context_window']:
        ema = ema[args['context_window']:-args['context_window']]
    return acoustic, ema


def load_utterance_list(set_name):
    """Load the list of utterances from a given set.

    set_name : name of the set, e.g. 'train', 'validation' or 'test'
    """
    with open(os.path.join(FOLDER, 'filesets', set_name + '.txt')) as file:
        return [row.strip('\n') for row in file]


def load_dataset(set_name, concatenate=False, **kwargs):
    """Load the acoustic and articulatory data of an entire set of utterances.

    set_name    : name of the set, e.g. 'train', 'validation' or 'test'
    concatenate : whether to concatenate the utterances into a single
                  numpy.ndarray (bool, default False)

    Any keyword argument of functions `load_acoustic` and `load_ema`
    may be passed ; otherwise, current setup values will be used.
    The latter can be seen and changed using `see_loading_setup`
    and `change_loading_setup`, all from the `data.mngu0.load` module.
    """
    fileset = load_utterance_list(set_name)
    dataset = np.array([load_utterance(name, **kwargs) for name in fileset])
    acoustic = np.array([utterance[0] for utterance in dataset])
    ema = np.array([utterance[1] for utterance in dataset])
    if concatenate:
        return np.concatenate(acoustic), np.concatenate(ema)
    return acoustic, ema
