# coding: utf-8

"""Set of functions to load pre-processed mngu0 data."""

import os

import numpy as np

from data.commons.enhance import build_context_windows
from data.mngu0.raw import get_utterances_list
from data.utils import CONSTANTS


FOLDER = CONSTANTS['mngu0_processed_folder']


_SETUP = {
    'audio_type': 'mfcc_stds',
    'context_window': 5,
    'dynamic_ema': False,
    'ema_norm': 'mean',
    'zero_padding': True
}


def change_loading_setup(
        audio_type=None, context_window=None, dynamic_ema=None,
        ema_norm=None, zero_padding=None
    ):
    """Update the default arguments used when importing mngu0 data.

    The parameters set through this function are those used by default
    by functions `load_utterance`, and `load_dataset` functions of the
    `data.mngu0.load` module.

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

    To see the current value of the parameters, use `see_loading_setup`.
    """
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
        name, audio_type='mfcc_stds', context_window=0, zero_padding=True
    ):
    """Load the acoustic data associated with an utterance from mngu0.

    name           : name of the utterance whose data to load (str)
    audio_type     : name of the audio features to use,
                     including normalization indications
                     (str, default 'mfcc_stds')
    context_window : half-size of the context window of frames to return
                     (default 0, returning single audio frames)
    zero_padding   : whether to zero-pad the data when building context
                     frames (bool, default True)
    """
    audio_type, norm_type = (audio_type + '_').split('_', 1)
    folder = (
        audio_type + '_norm_' + norm_type.strip('_')
        if norm_type else audio_type
    )
    path = os.path.join(FOLDER, folder, name + '_%s.npy' % audio_type)
    acoustic = np.load(path)
    if context_window:
        return build_context_windows(acoustic, context_window, zero_padding)
    return acoustic


def load_ema(name, norm_type='', use_dynamic=False):
    """Load the articulatory data associated with an utterance from mngu0.

    name           : name of the utterance whose data to load (str)
    norm_type      : optional type of normalization to use (str)
    use_dynamic    : whether to return dynamic features (bool, default False)
    """
    ema_dir = 'ema' if norm_type in ('', 'mean') else 'ema_norm_' + norm_type
    ema = np.load(os.path.join(FOLDER, ema_dir, name + '_ema.npy'))
    if norm_type == 'mean':
        ema -= get_normalization_parameters('ema')['global_means']
    if use_dynamic:
        return ema
    return ema[:, :ema.shape[1] // 3]


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
        name, args['audio_type'], args['context_window'], args['zero_padding']
    )
    ema = load_ema(name, args['ema_norm'], args['dynamic_ema'])
    if args['context_window'] and not args['zero_padding']:
        ema = ema[args['context_window']:-args['context_window']]
    return acoustic, ema


def get_utterances_set(set_name=None):
    """Get the list of utterances from a given set.

    set_name : name of the set, e.g. 'train', 'validation' or 'test'
    """
    if set_name is None:
        return get_utterances_list()
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
    fileset = get_utterances_set(set_name)
    dataset = np.array([load_utterance(name, **kwargs) for name in fileset])
    acoustic = np.array([utterance[0] for utterance in dataset])
    ema = np.array([utterance[1] for utterance in dataset])
    if concatenate:
        return np.concatenate(acoustic), np.concatenate(ema)
    return acoustic, ema
