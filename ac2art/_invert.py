# coding: utf-8

"""Function wrapping acoustic-to-articulatory inversion tasks."""

import os
import sys

import numpy as np

from ac2art.external.abkhazia import copy_feats, read_ark_file
from ac2art.networks import NeuralNetwork, load_dumped_model
from ac2art.utils import check_type_validity


def run_inversion(
        source, inverter, destination, keep_channels=None
    ):
    """Run acoustic-to-articulatory inversion of a set of features.

    Requires pre-computed acoustic features and a pre-trained
    acoustic-to-articulatory inverter neural network.

    source        : path to the **normalized** input features, which may
                    be stored as a single ark, scp or ark-like txt file,
                    or as npy files in a given folder
    inverter      : NeuralNetwork-inheriting instance, or path to
                    a .npy file recording a dumped model of such kind
    destination   : path where to output the inverted features, which
                    may be written as .npy files in a given folder or
                    compiled in a .ark, .scp or ark-like .txt file
    keep_channels : optional list of indexes of channel of inverted
                    features to keep (default None, implying all)
    """
    check_type_validity(source, str, 'source')
    check_type_validity(inverter, (NeuralNetwork, str), 'inverter')
    check_type_validity(destination, str, 'destination')
    # Set up a generator yielding inputs and a functions handling outputs.
    input_features = _read_features(source)
    handle_output = _setup_output_handler(destination)
    # Optionally load the inverter.
    if isinstance(inverter, str):
        print('Loading the inverter...')
        inverter = load_dumped_model(inverter)
    # Iteratively invert features and dump them to disk.
    for i, (utterance, input_data) in enumerate(input_features, 1):
        inverted_features = inverter.predict(input_data)
        if keep_channels:
            inverted_features = inverted_features[..., keep_channels]
        handle_output(utterance, inverted_features)
        print('Done inverting %s utterances.' % i)
        sys.stdout.write('\033[F')
    # When relevant, convert the output txt file to ark/scp files.
    if destination[-4:] in ('.ark', '.scp'):
        txt_file = destination[:-3] + 'txt'
        copy_feats(txt_file, destination)
        os.remove(txt_file)
    print('Done with the acoustic-to-articulatory inversion task.')


def _read_features(source):
    """Yield utterances' features, from npy files or an ark(-related) one."""
    # Handle the case of distinct .npy files.
    if os.path.isdir(source):
        return _read_npy_files(source)
    # Handle the case of single ark, scp or ark-like txt file.
    if os.path.isfile(source):
        return read_ark_file(source)
    # Raise exception if the referred source does not exist.
    raise FileNotFoundError(
        "Source folder or file '%s' does not exist." % source
    )


def _read_npy_files(folder):
    """Yield the contents of all .npy files in a given folder."""
    files = [name for name in os.listdir(folder) if name.endswith('.npy')]
    if not files:
        raise FileNotFoundError(
            "The '%s' folder does not contain any .npy file." % folder
        )
    return ((name, np.load(os.path.join(folder, name))) for name in files)


def _setup_output_handler(destination):
    """Set up and return a records storage function for inverted features."""
    # Handle the case when storing results as .npy files in a given folder.
    if not '.' in os.path.basename(destination):
        return __setup_npy_writer(destination)
    # Handle the case when storing results to a single ark(-related) file.
    extension = destination.rsplit('.', 1)[1]
    if extension in ('ark', 'scp', 'txt'):
        return __setup_ark_txt_writer(destination[:-3] + 'txt')
    # Raise exception if the argument points to an unsupported format.
    raise TypeError(
        'Invalid destination file extension: should be ark, txt or scp.'
    )


def __setup_npy_writer(folder):
    """Set up a records storage function to npy files in a given folder."""
    if not os.path.isdir(folder):
        os.makedirs(folder)

    def handle_output(utterance, features):
        """Store an utterance's data to a npy file."""
        nonlocal folder
        output_file = os.path.join(folder, utterance + '.npy')
        np.save(output_file, features)

    return handle_output


def __setup_ark_txt_writer(filename):
    """Set up a records storage function to a given ark-like txt file."""
    if os.path.isfile(filename):
        raise FileExistsError("File '%s' already exists." % filename)

    def handle_output(utterance, features):
        """Add an utterance's data to an ark-like txt file."""
        nonlocal filename
        string_array = '\n'.join(' '.join(map(str, row)) for row in features)
        with open(filename, mode='a', encoding='utf-8') as txt_file:
            txt_file.write(utterance + ' [\n' + string_array + ' ]\n')

    return handle_output
