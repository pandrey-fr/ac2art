# coding: utf-8

"""Set of functions to run ABXpy tasks on mngu0 data."""

import os
import functools

import h5features as h5f
import pandas as pd
import numpy as np

from data.commons.abxpy import abxpy_pipeline, abxpy_task
from data.mngu0.raw import load_phone_labels
from data.mngu0.load import load_acoustic, load_ema, get_utterances_set
from data.utils import check_positive_int, check_type_validity, CONSTANTS


ABX_FOLDER = os.path.join(CONSTANTS['mngu0_processed_folder'], 'abx')


def extract_h5_features(
        audio_features=None, ema_features=None, inverter=None,
        output_name='mngu0_features', dynamic_ema=True, sampling_rate=200
    ):
    """Build an h5 file recording audio features associated with mngu0 data.

    audio_features : optional name of audio features to use, including
                     normalization indications
    ema_features   : optional name of ema features' normalization to use
                     (use '' for raw data and None for no EMA data)
    inverter       : optional acoustic-articulatory inverter whose predictions
                     to return, based on the specified audio features
    output_name    : base name of the output file (default 'mngu0_features')
    dynamic_ema    : whether to include dynamic articulatory features
                     (bool, default True)
    sampling_rate  : sampling rate of the frames, in Hz (int, default 200)
    """
    # Check that the destination file does not exist.
    output_file = os.path.join(ABX_FOLDER, '%s.features' % output_name)
    if os.path.isfile(output_file):
        raise FileExistsError("File '%s' already exists." % output_file)
    # Set up the features loading function.
    load_features = _setup_features_loader(
        audio_features, ema_features, inverter, dynamic_ema
    )
    # Load the list of utterances and process them iteratively.
    utterances = get_utterances_set()
    with h5f.Writer(output_file) as writer:
        for i in range(0, len(utterances), 100):
            # Load or compute the utterances list, features and time labels.
            items = utterances[i:i + 100]
            features = [load_features(item) for item in items]
            labels = [np.arange(len(data)) / sampling_rate for data in features]
            # Write the currently processed utterances' data to h5.
            writer.write(
                h5f.Data(items, labels, features, check=True),
                groupname='features', append=True
            )


def _setup_features_loader(audio_features, ema_features, inverter, dynamic_ema):
    """Build a function to load features associated with an mngu0 utterance.

    See `data.mngu0.abx.extract_h5_features` documentation for arguments.
    """
    if audio_features is None and ema_features is None:
        raise RuntimeError('No features were set to be included.')
    if inverter is not None:
        if audio_features is None:
            raise RuntimeError(
                'No acoustic features specified to feed the inverter.'
            )
        elif ema_features is not None:
            raise RuntimeError(
                'Both articulatory features and an inverter were specified.'
            )
    # Build the acoustic features loading function.
    if audio_features is not None:
        window = 0 if inverter is None or inverter.input_shape[1] % 11 else 5
        load_audio = functools.partial(
            load_acoustic, audio_type=audio_features, context_window=window
        )
        # Optionally build and return an inverter-based features loader.
        if inverter is not None:
            def invert_features(utterance):
                """Return the features inverted from an utterance."""
                return inverter.predict(load_audio(utterance))
            return invert_features
        elif ema_features is None:
            return load_audio
    # Build the articulatory features loading function.
    if ema_features is not None:
        load_articulatory = functools.partial(
            load_ema, norm_type=ema_features, use_dynamic=dynamic_ema
        )
        if audio_features is None:
            return load_articulatory
    # When appropriate, build a global features loading function.
    def load_features(utterance):
        """Load the features associated with an utterance."""
        return np.concatenate(
            [load_audio(utterance), load_articulatory(utterance)], axis=1
        )
    return load_features


def make_itemfile(dataset=None):
    """Build a .item file for ABXpy recording mngu0 phone labels.

    dataset : optional set name whose utterances to use (str)
    """
    utterances = get_utterances_set(dataset)
    name = 'mngu0_%sphones.item' % ('' if dataset is None else dataset + '_')
    output_file = os.path.join(ABX_FOLDER, name)
    columns = ['#file', 'onset', 'offset', '#phone', 'context']
    with open(output_file, mode='w') as itemfile:
        itemfile.write(' '.join(columns) + '\n')
    for utterance in utterances:
        items = pd.DataFrame(_phones_to_itemfile(utterance))
        items[columns].to_csv(
            output_file, index=False, header=False,
            sep=' ', mode='a', encoding='utf-8'
        )


def _phones_to_itemfile(utterance):
    """Build a dict of item file rows for a given mngu0 utterance."""
    phones = load_phone_labels(utterance)
    times = [round(time - phones[0][0], 3) for time, _ in phones[:-1]]
    phones = [phone for _, phone in phones]
    return {
        '#file': [utterance] * (len(times) - 1),
        'onset': times[:-1],
        'offset': times[1:],
        '#phone': phones[1:-1],
        'context': [
            phones[i - 1] + '_' + phones[i + 1] for i in range(1, len(times))
        ]
    }


def make_abx_task(dataset=None):
    """Build a .abx ABXpy task file associated with mngu0 phones.

    dataset : optional set name whose utterances to use (str)
    """
    # Build the item file if necessary.
    extension = '' if dataset is None else dataset + '_'
    item_file = os.path.join(ABX_FOLDER, 'mngu0_%sphones.item' % extension)
    if not os.path.isfile(item_file):
        make_itemfile(dataset)
    # Run the ABXpy task module.
    output = os.path.join(ABX_FOLDER, 'mngu0_%stask.abx' % extension)
    abxpy_task(item_file, output, on='phone', by='context')


def abx_from_features(features_filename, dataset=None, n_jobs=1):
    """Run the ABXpy pipeline on a set of mngu0 features.

    features_file : name of a h5 file of mngu0 features created with
                    the `extract_h5_features` function (str)
    dataset       : optional set name whose utterances' features are used (str)
    n_jobs        : number of CPU cores to use (positive int, default 1)
    """
    check_type_validity(features_filename, str, 'features_filename')
    check_type_validity(dataset, (str, type(None)), 'dataset')
    check_positive_int(n_jobs, 'n_jobs')
    # Declare paths to the files used.
    task_file = 'mngu0_%stask.abx' % ('' if dataset is None else dataset + '_')
    task_file = os.path.join(ABX_FOLDER, task_file)
    features_file = os.path.join(ABX_FOLDER, features_filename + '.features')
    extension = '%s_abx.csv' % ('' if dataset is None else '_' + dataset)
    output_file = os.path.join(ABX_FOLDER, features_filename + extension)
    # Check that the features file exists.
    if not os.path.exists(features_file):
        raise FileNotFoundError("File '%s' does not exist." % features_file)
    # Build the ABX task file if necessary.
    if not os.path.isfile(task_file):
        make_abx_task(dataset)
    # Run the ABXpy pipeline.
    abxpy_pipeline(features_file, task_file, output_file, n_jobs)
