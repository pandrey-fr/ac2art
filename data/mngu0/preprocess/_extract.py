# coding: utf-8

"""Set of functions to extract raw mngu0 data."""

import os
import sys
import time

import numpy as np
import resampy

from data.mngu0.raw import (
    get_utterances_list, load_ema, load_phone_labels, load_wav
)
from data.utils import check_positive_int, check_type_validity, load_data_paths


RAW_FOLDER, NEW_FOLDER = load_data_paths('mngu0')


def adjust_filesets():
    """Adjust the filesets lists provided with mngu0 to fit the raw data.

    As the provided filesets lists are based on a extracted version of
    the mngu0 corpus anterior to the splitting of some of the utterances
    within the raw data, running this function is necessary so as to use
    similar train, validation and test utterances of newly-processed data
    as in most papers using this database.
    """
    # Load the full list of utterances.
    utterances = get_utterances_list()
    # Build the output 'filesets' folder if needed.
    output_folder = os.path.join(NEW_FOLDER, 'filesets')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # Iterate over the three filesets to produce.
    for set_name in ['train', 'validation', 'test']:
        # Load the raw fileset list.
        path = os.path.join(RAW_FOLDER, 'ema_filesets', set_name + 'files.txt')
        with open(path) as file:
            raw_fileset = [row.strip('\n') for row in file]
        # Derive the correct fileset of newly processed utterances.
        new_fileset = [
            utterance for utterance in utterances
            if utterance.strip('abcdef') in raw_fileset
        ]
        # Write the adjusted fileset to a txt file.
        with open(os.path.join(output_folder, set_name + '.txt'), 'w') as file:
            file.write('\n'.join(new_fileset))


DOC_EXTRACT_DETAILS = """
    The extractations include the following:
      - optional resampling of the EMA data
      - framing of audio data to align acoustic and articulatory records
      - production of various acoustic features based on the audio data
      - trimming of silences at the beginning and end of each utterance
"""


DOC_EXTRACT_ARGUMENTS = """
    n_coeff           : number of coefficients to compute for each
                        representation of the audio data - either
                        a single int or a tuple of ints (default 40)
    audio_forms       : optional list of representations of the audio data
                        to produce, among {'energy', 'lsf', 'lpc', 'mfcc'}
                        (list of str, default None implying all of them)
    articulators_list : optional list of raw EMA data columns to keep
                        (default None, implying twelve, detailed below)
    ema_sampling_rate : sample rate of the EMA data to use, in Hz
                        (int, default 200)
    audio_frames_size : number of acoustic samples to include per frame
                        (int, default 200)

    The default values of the last three arguments echo those used in most
    papers making use of the mngu0 data. The articulators kept correspond
    to the front-to-back and top-to-bottom (resp. py and pz) coordinates of
    six articulators : tongue tip (T1), tongue dorsum (T2), tongue back (T3),
    jaw, upperlip and lowerlip.
"""


def extract_all_utterances(
        n_coeff=40, audio_forms=None, articulators_list=None,
        ema_sampling_rate=200, audio_frames_size=200
    ):
    """Extract acoustic and articulatory data of each mngu0 utterance.
    {0}
    The produced data is stored to the 'mngu0_processed_folder' set in the
    json configuration file, where a subfolder is built for each kind of
    features (ema, mfcc, etc.). Each utterance is stored as a '.npy' file.
    The file names include the utterance's name, extended with an indicator
    of the kind of features it contains.
    {1}
    """
    # Check arguments validity, assign default values and build output folders.
    audio_forms, articulators_list = scan_extractation_parameters(
        n_coeff, audio_forms, articulators_list,
        ema_sampling_rate, audio_frames_size
    )
    # Iterate over all mngu0 utterances.
    for utterance in get_utterances_list():
        _extract_utterance_data(
            utterance, n_coeff, audio_forms, articulators_list,
            ema_sampling_rate, audio_frames_size
        )
        end_time = time.asctime().split(' ')[-2]
        print('%s : Done with utterance %s.' % (end_time, utterance))
        sys.stdout.write('\033[F')


extract_all_utterances.__doc__ = extract_all_utterances.__doc__.format(
    DOC_EXTRACT_DETAILS, DOC_EXTRACT_ARGUMENTS
)


def extract_utterance_data(
        utterance, n_coeff=40, audio_forms=None, articulators_list=None,
        ema_sampling_rate=200, audio_frames_size=200
    ):
    """Extract acoustic and articulatory data of a given mngu0 utterance.
    {0}
    The produced data is stored to a subfolder of 'mngu0_processed_folder'
    set in the json configuration file, named after the kind of features
    produced (ema, lsf...). The utterance data is stored as a '.npy' file.
    The file name is the utterance's name, extended with an indicator of
    the kind of features it contains.

    utterance         : name of the utterance to process (str){1}
    """
    # Check arguments validity, assign default values and build output folders.
    check_type_validity(utterance, str, 'utterance')
    audio_forms, articulators_list = scan_extractation_parameters(
        n_coeff, audio_forms, articulators_list,
        ema_sampling_rate, audio_frames_size
    )
    # Conduct the actual data extractation.
    _extract_utterance_data(
        utterance, n_coeff, audio_forms, articulators_list,
        ema_sampling_rate, audio_frames_size
    )


extract_utterance_data.__doc__ = extract_utterance_data.__doc__.format(
    DOC_EXTRACT_DETAILS, DOC_EXTRACT_ARGUMENTS
)


def _extract_utterance_data(
        utterance, n_coeff, audio_forms, articulators_list,
        ema_sampling_rate, audio_frames_size
    ):
    """Extract acoustic and articulatory data of a given mngu0 utterance.

    This function serves as a dependency for `extract_all_utterances`
    and `extract_utterance_data`, avoiding to check again and again
    the same arguments when using the former while ensuring arguments
    are being checked when using the latter.
    """
    # Load phone labels and compute frames index so as to trim silences.
    labels = load_phone_labels(utterance)
    start_frame = int(
        (labels[0][0] if labels[0][1] == '#' else 0) * ema_sampling_rate
    )
    end_frame = int(
        (labels[-2][0] if labels[-1][1] == '#' else labels[-1][0])
        * ema_sampling_rate
    )
    # Load EMA data and optionally resample it.
    ema, _ = load_ema(utterance, articulators_list)
    if ema_sampling_rate != 200:
        ema = resampy.resample(
            ema, sr_orig=200, sr_new=ema_sampling_rate, axis=0
        )
    # Trim silences off the EMA data and save it to disk.
    path = os.path.join(NEW_FOLDER, 'ema', utterance + '_ema.npy')
    np.save(path, ema[start_frame:end_frame])
    # Load the audio waveform data, structuring it into frames.
    wav = load_wav(
        utterance, audio_frames_size, hop_time=int(1000 / ema_sampling_rate)
    )
    # Compute each representation of the audio, trim its silences and save it.
    for name in audio_forms:
        audio = (
            getattr(wav, 'get_' + name)(n_coeff) if name != 'energy'
            else wav.get_rms_energy()
        )
        audio = audio[start_frame:end_frame]
        path = os.path.join(NEW_FOLDER, name, utterance + '_%s.npy' % name)
        np.save(path, audio)


def scan_extractation_parameters(
        n_coeff, audio_forms, articulators_list, ema_sampling_rate,
        audio_frames_size
    ):
    """Check the validity of arguments provided to process a mngu0 utterance.

    Build the necessary subfolders so as to store the processed data.

    Replace `audio_forms` and `articulators_list` with their default values
    when they are passed as None.

    Return the definitive values of the latter two arguments.
    """
    # Check positive integer arguments.
    check_positive_int(n_coeff, 'n_coeff')
    check_positive_int(ema_sampling_rate, 'ema_sampling_rate')
    if 1000 % ema_sampling_rate:
        raise ValueError("'ema_sampling_rate' must be a divisor of 1000.")
    check_positive_int(audio_frames_size, 'audio_frames_size')
    # Check audio_forms argument validity.
    _audio_forms = ['energy', 'lpc', 'lsf', 'mfcc']
    if audio_forms is None:
        audio_forms = _audio_forms
    else:
        check_type_validity(audio_forms, list, 'audio_forms')
        invalid = [name for name in audio_forms if name not in _audio_forms]
        if invalid:
            raise ValueError("Unknown audio representation(s): %s." % invalid)
    # Build necessary folders to store the processed data.
    for name in audio_forms + ['ema']:
        dirname = os.path.join(NEW_FOLDER, name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
    # Check articulators_list argument validity.
    if articulators_list is None:
        articulators_list = [
            'T3_py', 'T3_pz', 'T2_py', 'T2_pz', 'T1_py', 'T1_pz',
            'jaw_py', 'jaw_pz', 'upperlip_py', 'upperlip_pz',
            'lowerlip_py', 'lowerlip_pz'
        ]
    else:
        check_type_validity(articulators_list, list, 'articulators_list')
    # Return potentially altered list arguments.
    return audio_forms, articulators_list
