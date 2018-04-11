# coding: utf-8

"""Wrappers to design corpus-specific functions to run ABXpy tasks."""

import os
import functools

import h5features as h5f
import pandas as pd
import numpy as np

from data.commons.abxpy import abxpy_pipeline, abxpy_task
from data.utils import (
    check_positive_int, check_type_validity, CONSTANTS, import_from_string
)


def build_h5features_extractor(corpus):
    """Define and return a function extracting features to h5 files."""
    # Load dependency path and functions.
    abx_folder = os.path.join(CONSTANTS['%s_processed_folder' % corpus], 'abx')
    load_acoustic, load_ema, get_utterances = import_from_string(
        'data.%s.load._load' % corpus,
        ['load_acoustic', 'load_ema', 'get_utterances']
    )
    # Define features extraction functions.

    def _setup_features_loader(
            audio_features, ema_features, inverter, dynamic_ema
        ):
        """Build a function to load features associated with an utterance.

        See `extract_h5_features` documentation for arguments.
        """
        nonlocal load_acoustic, load_ema
        # Check that provided arguments make sense.
        if audio_features is None and ema_features is None:
            raise RuntimeError('No features were set to be included.')
        if inverter is not None:
            if audio_features is None:
                raise RuntimeError(
                    'No acoustic features specified to feed the inverter.'
                )
            elif ema_features is not None:
                raise RuntimeError(
                    'Both ema features and an inverter were specified.'
                )
        # Build the acoustic features loading function.
        if audio_features is not None:
            window = (
                0 if inverter is None or inverter.input_shape[1] % 11 else 5
            )
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

    def extract_h5_features(
            audio_features=None, ema_features=None, inverter=None,
            output_name='%s_features' % corpus, dynamic_ema=True,
            sampling_rate=200
        ):
        """Build an h5 file recording audio features associated with {0} data.

        audio_features : optional name of audio features to use, including
                         normalization indications
        ema_features   : optional name of ema features' normalization to use
                         (use '' for raw data and None for no EMA data)
        inverter       : optional acoustic-articulatory inverter whose
                         predictions to use, based on the audio features
        output_name    : base name of the output file (default '{0}_features')
        dynamic_ema    : whether to include dynamic articulatory features
                         (bool, default True)
        sampling_rate  : sampling rate of the frames, in Hz (int, default 200)
        """
        nonlocal abx_folder, get_utterances, _setup_features_loader
        # Check that the destination file does not exist.
        output_file = os.path.join(abx_folder, '%s.features' % output_name)
        if os.path.isfile(output_file):
            raise FileExistsError("File '%s' already exists." % output_file)
        # Set up the features loading function.
        load_features = _setup_features_loader(
            audio_features, ema_features, inverter, dynamic_ema
        )
        # Load the list of utterances and process them iteratively.
        utterances = get_utterances()
        with h5f.Writer(output_file) as writer:
            for i in range(0, len(utterances), 100):
                # Load or compute utterances list, features and time labels.
                items = utterances[i:i + 100]
                features = [load_features(item) for item in items]
                labels = [
                    np.arange(len(data)) / sampling_rate for data in features
                ]
                # Write the currently processed utterances' data to h5.
                writer.write(
                    h5f.Data(items, labels, features, check=True),
                    groupname='features', append=True
                )

    # Adjust the features extraction function's docstring and return it.
    extract_h5_features.__doc__ = extract_h5_features.__doc__.format(corpus)
    return extract_h5_features


def build_abxpy_callers(corpus):
    """Define and return corpus-specific functions to run ABXpy tasks."""
    # Load dependency path and function.
    abx_folder = os.path.join(CONSTANTS['%s_processed_folder' % corpus], 'abx')
    get_utterances = import_from_string(
        'data.%s.load._load' % corpus, 'get_utterances'
    )
    load_phone_labels = import_from_string(
        'data.%s.raw._loaders' % corpus, 'load_phone_labels'
    )
    # Define the functions.

    def _phones_to_itemfile(utterance):
        """Build a dict of item file rows for a given utterance."""
        nonlocal load_phone_labels
        phones = load_phone_labels(utterance)
        times = [round(time - phones[0][0], 3) for time, _ in phones[:-1]]
        phones = [phone for _, phone in phones]
        return {
            '#file': [utterance] * (len(times) - 1),
            'onset': times[:-1],
            'offset': times[1:],
            '#phone': phones[1:-1],
            'context': [
                phones[i - 1] + '_' + phones[i + 1]
                for i in range(1, len(times))
            ]
        }

    def make_itemfile(fileset=None):
        """Build a .item file for ABXpy recording {0} phone labels.

        fileset : optional set name whose utterances to use (str)
        """
        nonlocal abx_folder, corpus, get_utterances, _phones_to_itemfile
        utterances = get_utterances(fileset)
        name = '%s_%sphones.item' % (
            corpus, '' if fileset is None else fileset + '_'
        )
        output_file = os.path.join(abx_folder, name)
        columns = ['#file', 'onset', 'offset', '#phone', 'context']
        with open(output_file, mode='w') as itemfile:
            itemfile.write(' '.join(columns) + '\n')
        for utterance in utterances:
            items = pd.DataFrame(_phones_to_itemfile(utterance))
            items[columns].to_csv(
                output_file, index=False, header=False,
                sep=' ', mode='a', encoding='utf-8'
            )

    def make_abx_task(fileset=None):
        """Build a .abx ABXpy task file associated with {0} phones.

        fileset : optional set name whose utterances to use (str)
        """
        nonlocal abx_folder, corpus, make_itemfile
        # Build the item file if necessary.
        name = '%s_%s' % (corpus, '' if fileset is None else fileset + '_')
        item_file = os.path.join(abx_folder, '%sphones.item' % name)
        if not os.path.isfile(item_file):
            make_itemfile(fileset)
        # Run the ABXpy task module.
        output = os.path.join(abx_folder, '%stask.abx' % name)
        abxpy_task(item_file, output, on='phone', by='context')

    def abx_from_features(features_filename, fileset=None, n_jobs=1):
        """Run the ABXpy pipeline on a set of pre-extracted {0} features.

        features_file : name of a h5 file of {0} features created with
                        the `extract_h5_features` function (str)
        fileset       : optional name of a fileset whose utterances'
                        features to use (str)
        n_jobs        : number of CPU cores to use (positive int, default 1)
        """
        nonlocal abx_folder, corpus, make_abx_task
        check_type_validity(features_filename, str, 'features_filename')
        check_type_validity(fileset, (str, type(None)), 'fileset')
        check_positive_int(n_jobs, 'n_jobs')
        # Declare paths to the files used.
        task_file = '%s_%stask.abx' % (
            corpus, '' if fileset is None else fileset + '_'
        )
        task_file = os.path.join(abx_folder, task_file)
        features_file = os.path.join(
            abx_folder, features_filename + '.features'
        )
        extension = '%s_abx.csv' % ('' if fileset is None else '_' + fileset)
        output_file = os.path.join(abx_folder, features_filename + extension)
        # Check that the features file exists.
        if not os.path.exists(features_file):
            raise FileNotFoundError("No such file: '%s'." % features_file)
        # Build the ABX task file if necessary.
        if not os.path.isfile(task_file):
            make_abx_task(fileset)
        # Run the ABXpy pipeline.
        abxpy_pipeline(features_file, task_file, output_file, n_jobs)

    def load_abx_scores(filename):
        """Load, aggregate and return some pre-computed abx scores."""
        nonlocal abx_folder
        path = os.path.join(abx_folder, filename + '_abx.csv')
        data = pd.read_csv(path, sep='\t')
        data['score'] *= data['n']
        data['phones'] = data.apply(
            lambda row: '_'.join(sorted([row['phone_1'], row['phone_2']])),
            axis=1
        )
        scores = data.groupby('phones')[['score', 'n']].sum()
        scores['score'] /= scores['n']
        return scores

    # Adjust functions' docstrings and return them.
    functions = (
        abx_from_features, make_abx_task, make_itemfile, load_abx_scores
    )
    for function in functions:
        function.__doc__ = function.__doc__.format(corpus)
    return functions
