# coding: utf-8

"""Wrapper building cropus-specific data extraction functions."""

import os
import sys
import time

import numpy as np
import scipy.signal

from ac2art.external.abkhazia import (
    ark_to_npy, compute_mfcc, prepare_abkhazia_corpus
)
from ac2art.internal.data_utils import interpolate_missing_values
from ac2art.utils import (
    check_positive_int, check_type_validity, import_from_string, CONSTANTS
)


def build_features_extraction_functions(
        corpus, initial_sampling_rate, default_articulators, docstring_details
    ):
    """Define and return a raw features extraction function for a corpus.

    corpus                : name of the corpus whose features to extract (str)
    initial_sampling_rate : initial sampling rate of the EMA data, in Hz (int)
    default_articulators  : default articulators to keep (list of str)
    docstring_details     : docstring complement for the returned functions

    Return a single function:
      - extract_utterances_data
    """
    # Long but explicit function name; pylint: disable=invalid-name
    # Define auxiliary functions through wrappers.
    control_arguments = build_arguments_checker(corpus, default_articulators)
    extract_data = build_extractor(corpus, initial_sampling_rate)
    # Import the get_utterances_list dependency function.
    get_utterances_list = import_from_string(
        'ac2art.corpora.%s.raw._loaders' % corpus, 'get_utterances_list'
    )
    # Define a function extracting features from all utterances.
    def extract_utterances_data(
            audio_forms=None, n_coeff=13, articulators_list=None,
            ema_sampling_rate=100, audio_frames_time=25
        ):
        """Extract acoustic and articulatory data of each {0} utterance.

        audio_forms       : optional list of representations of the audio data
                            to produce, among {{'lsf', 'lpc', 'mfcc'}}
                            (list of str, default None implying all of them)
        n_coeff           : number of static coefficients to compute for each
                            representation of the audio data, either as a
                            single int or a list of int (default 13)
                            Note : dynamic features will be added to those.
        articulators_list : optional list of raw EMA data columns to keep
                            (default None, implying twelve, detailed below)
        ema_sampling_rate : sample rate of the EMA data to use, in Hz
                            (int, default 100)
        audio_frames_time : duration of the audio frames used to compute
                            acoustic features, in milliseconds
                            (int, default 25)

        Data extractation includes the following:
          - optional resampling of the EMA data
          - framing of audio data to align acoustic and articulatory records
          - production of various acoustic features based on the audio data
          - trimming of silences at the beginning and end of each utterance

        Note: 'mfcc' audio form will produce MFCC coefficients enriched
              with pitch features, computed using abkhazia. Alternative
              computation using `data.commons.loaders.Wav.get_mfcc()` may
              be obtained using the 'mfcc_' keyword instead.

        The produced data is stored to the '{0}_processed_folder' set in the
        json configuration file, where a subfolder is built for each kind of
        features (ema, mfcc, etc.). Each utterance is stored as a '.npy' file.
        The file names include the utterance's name, extended with an indicator
        of the kind of features it contains.

        {1}
        """
        nonlocal corpus, control_arguments, extract_data, get_utterances_list
        # Check arguments, assign default values and build output folders.
        audio_forms, n_coeff, articulators_list = control_arguments(
            audio_forms, n_coeff, articulators_list,
            ema_sampling_rate, audio_frames_time
        )
        # Compute mfcc coefficients using abkhazia, if relevant.
        abkhazia_mfcc = 'mfcc' in audio_forms
        if abkhazia_mfcc:
            mfcc_ix = audio_forms.index('mfcc')
            wav_to_mfcc(
                corpus, n_coeff=n_coeff[mfcc_ix], pitch=True,
                frame_time=audio_frames_time,
                hop_time=(1000 / ema_sampling_rate)
            )
            del audio_forms[mfcc_ix]
            del n_coeff[mfcc_ix]
        # Iterate over all corpus utterances.
        for utterance in get_utterances_list():
            extract_data(
                utterance, audio_forms, n_coeff, abkhazia_mfcc,
                articulators_list, ema_sampling_rate, audio_frames_time
            )
            end_time = time.asctime().split(' ')[-2]
            print('%s : Done with utterance %s.' % (end_time, utterance))
            sys.stdout.write('\033[F')
        # Record the list of articulators.
        path = os.path.join(
            CONSTANTS['%s_processed_folder' % corpus], 'ema', 'articulators'
        )
        with open(path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(articulators_list))

    # Adjust the function's docstring and return it.
    extract_utterances_data.__doc__ = (
        extract_utterances_data.__doc__.format(corpus, docstring_details)
    )
    return extract_utterances_data


def build_arguments_checker(corpus, default_articulators):
    """Define and return a function checking features extraction arguments."""
    new_folder = CONSTANTS['%s_processed_folder' % corpus]

    def control_arguments(
            audio_forms, n_coeff, articulators_list,
            ema_sampling_rate, audio_frames_time
        ):
        """Control the arguments provided to extract some features.

        Build the necessary subfolders so as to store the processed data.

        Replace `audio_forms` and `articulators_list` with their
        default values when they are passed as None.

        Return the definitive values of the latter two arguments.
        """
        nonlocal default_articulators, new_folder
        # Check positive integer arguments.
        check_positive_int(ema_sampling_rate, 'ema_sampling_rate')
        check_positive_int(audio_frames_time, 'audio_frames_time')
        # Check audio_forms argument validity.
        valid_forms = ['lpc', 'lsf', 'mfcc', 'mfcc_']
        if audio_forms is None:
            audio_forms = valid_forms[:-1]
        else:
            if isinstance(audio_forms, str):
                audio_forms = [audio_forms]
            elif isinstance(audio_forms, tuple):
                audio_forms = list(audio_forms)
            else:
                check_type_validity(audio_forms, list, 'audio_forms')
            invalid = [name for name in audio_forms if name not in valid_forms]
            if invalid:
                raise ValueError(
                    "Unknown audio representation(s): %s." % invalid
                )
        # Check n_coeff argument validity.
        check_type_validity(n_coeff, (int, tuple, list), 'n_coeff')
        if isinstance(n_coeff, int):
            check_positive_int(n_coeff, 'single `n_coeff` value')
            n_coeff = [n_coeff] * len(audio_forms)
        elif len(n_coeff) != len(audio_forms):
            raise TypeError(
                "'n_coeff' sequence should be of same length as 'audio_forms'."
            )
        # Build necessary folders to store the processed data.
        for name in audio_forms + ['ema', 'voicing']:
            dirname = os.path.join(new_folder, name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        # Check articulators_list argument validity.
        if articulators_list is None:
            articulators_list = default_articulators
        else:
            check_type_validity(articulators_list, list, 'articulators_list')
        # Return potentially altered list arguments.
        return audio_forms, n_coeff, articulators_list

    # Return the defined function.
    return control_arguments


def build_extractor(corpus, initial_sampling_rate):
    """Define and return a function to extract features from an utterance.

    corpus                : name of the corpus from which to import features
    initial_sampling_rate : initial sampling rate of the EMA data, in Hz (int)
    """
    # Load the output path and dependency data loading functions.
    new_folder = CONSTANTS['%s_processed_folder' % corpus]
    load_ema, load_phone_labels, load_voicing, load_wav = import_from_string(
        module='ac2art.corpora.%s.raw._loaders' % corpus,
        elements=['load_ema', 'load_phone_labels', 'load_voicing', 'load_wav']
    )

    def get_boundaries(utterance, sampling_rate):
        """Return frames index to use so as to trim edge silences."""
        nonlocal load_phone_labels
        # Load phone labels and gather edge silences' timecodes.
        labels = load_phone_labels(utterance)
        start_time = labels[0][0] if labels[0][1] == '#' else 0
        end_time = labels[-2][0] if labels[-1][1] == '#' else labels[-1][0]
        # Compute and return associated frame indexes.
        start_frame = int(np.floor(start_time * sampling_rate))
        end_frame = int(np.ceil(end_time * sampling_rate))
        return start_frame, end_frame

    def extract_ema(
            utterance, sampling_rate, articulators
        ):
        """Extract and return the EMA data associated with an utterance."""
        nonlocal initial_sampling_rate, load_ema, new_folder
        # Load EMA data and interpolate NaN values using cubic splines.
        ema, _ = load_ema(utterance, articulators)
        ema = np.concatenate([
            interpolate_missing_values(data_column).reshape(-1, 1)
            for data_column in np.transpose(ema)
        ], axis=1)
        # Optionally resample the EMA data.
        if sampling_rate != initial_sampling_rate:
            ratio = sampling_rate / initial_sampling_rate
            ema = scipy.signal.resample(ema, num=int(len(ema) * ratio))
        # Return the EMA data.
        return ema

    def extract_audio(
            utterance, audio_forms, n_coeff, sampling_rate, frames_time
        ):
        """Generate and return speech features for an utterance."""
        # Wrapped function; pylint: disable=too-many-arguments
        nonlocal corpus, load_wav, new_folder
        hop_time = (1000 / sampling_rate)
        wav = load_wav(utterance, frames_time, hop_time)
        return {
            name: wav.get(name.strip('_'), n_feat, static_only=False)
            for name, n_feat in zip(audio_forms, n_coeff)
        }

    def extract_data(
            utterance, audio_forms, n_coeff, abkhazia_mfcc,
            articulators, sampling_rate, frames_time
        ):
        """Extract acoustic and articulatory data of a given utterance."""
        # Wrapped function; pylint: disable=too-many-arguments
        nonlocal load_wav, new_folder
        nonlocal extract_audio, extract_ema, get_boundaries
        # Generate or load all kinds of features for the utterance.
        data = extract_audio(
            utterance, audio_forms, n_coeff, sampling_rate, frames_time
        )
        data['ema'] = extract_ema(utterance, sampling_rate, articulators)
        data['voicing'] = load_voicing(utterance, sampling_rate)
        if abkhazia_mfcc:
            path = os.path.join(new_folder, 'mfcc', utterance + '.npy')
            data['mfcc'] = np.load(path)
        # Fit the edge silences trimming values.
        start_frame, end_frame = get_boundaries(utterance, sampling_rate)
        original_end = end_frame
        for name, array in data.items():
            length = len(array)
            if length < start_frame:
                raise ValueError(
                    "Utterance '%s': '%s' features are shorter than the"
                    "expected start trimming zone." % (utterance, name)
                )
            if length < original_end:
                print(
                    "Utterance '%s': '%s' features are shorter than expected"
                    "(%s vs %s).\nAll features will be trimmed to fit."
                    % (utterance, name, length, original_end)
                )
                if length < end_frame:
                    end_frame = length
        # Trim and save all features sets to disk.
        for name, array in data.items():
            path = os.path.join(
                new_folder, name, utterance + '_' + name + '.npy'
            )
            np.save(path, array[start_frame:end_frame])

    # Return the previous last function.
    return extract_data


def wav_to_mfcc(
        corpus, n_coeff=13, pitch=True, frame_time=25, hop_time=10
    ):
    """Produce MFCC features using abkhazia for a given corpus."""
    print('Running MFCC computation with abkhazia...')
    # Establish folders to work with.
    main_folder = CONSTANTS['%s_processed_folder' % corpus]
    data_folder = os.path.join(main_folder, 'abkhazia', 'data')
    mfcc_folder = os.path.join(main_folder, 'abkhazia', 'mfcc_features')
    output_folder = os.path.join(main_folder, 'mfcc')
    # Build an abkhazia data folder for the corpus.
    prepare_abkhazia_corpus(corpus, data_folder)
    # Compute MFCC features using abkhazia.
    compute_mfcc(
        data_folder, n_coeff, pitch, frame_time, hop_time, mfcc_folder
    )
    # Extract MFCC features to utterance-wise npy files.
    ark_to_npy(os.path.join(mfcc_folder, 'feats.scp'), output_folder)
    print('Done producing raw MFCC coefficients with abkhazia.')
