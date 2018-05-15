# coding: utf-8

"""Set of utility functions to reformat data used with abkhazia."""

import os

import numpy as np

from data.utils import CONSTANTS
from utils import check_type_validity


def ark_to_txt(ark_file, output_file=None):
    """Copy data from a (set of) ark file(s) to a txt file.

    ark_file    : path to a .ark file or to a .scp file indexing
                  data from one or more .ark files
    output_file : optional output file path - existing files
                  will be overwritten (default: .txt file of
                  same name and location as `ark_file`)
    """
    # Check the arguments' validity.
    check_type_validity(ark_file, str, 'ark_file')
    check_type_validity(output_file, (str, type(None)), 'output_file')
    if not os.path.exists(ark_file):
        raise FileNotFoundError("'%s' does not exist." % ark_file)
    file_type = ark_file.rsplit('.', 1)[1]
    if file_type not in ('ark', 'scp'):
        raise TypeError("'ark_file' should point to a .ark or .scp file.")
    # Set up the copy-feats command.
    if output_file is None:
        output_file = ark_file[:-4] + '.txt'
    command = os.path.join(CONSTANTS['kaldi_folder'], 'src/featbin/copy-feats')
    command += ' %s:%s t,ark:%s' % (file_type, ark_file, output_file)
    # Run ark-to-txt conversion.
    status = os.system(command)
    if status != 0:
        raise RuntimeError(
            'kaldi copy-feats (ark to txt) exited with error code %s.' % status
        )
    print('Successfully ran ark to txt conversion with kaldi.')


def ark_to_npy(ark_file, output_folder):
    """Extract utterances data from a (set of) ark file(s) to npy files.

    ark_file      : ark file whose data to extract - may be a .ark
                    file, a .scp file indexing one or more .ark files
                    or a .txt file created from one of the previous
    output_folder : folder in which to extract utterance-wise npy files
                    (note that existing files will be overwritten)
    """
    # Check parameters validity. Build output folder if needed.
    check_type_validity(ark_file, str, 'ark_file')
    check_type_validity(output_folder, str, 'output_folder')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # If needed, convert data from ark to txt.
    if ark_file.endswith('.ark') or ark_file.endswith('.scp'):
        ark_to_txt(ark_file, ark_file[:-3] + 'txt')
        ark_file = ark_file[:-3] + 'txt'
        temp_txt = True
    elif not ark_file.endswith('.txt'):
        raise TypeError("'ark_file' extension should be .ark or .txt file.")
    else:
        temp_txt = False
    # Iteratively extract utterances' data and write individual npy files.
    with open(ark_file) as source_file:
        utterance = None
        array = []
        n_files = 0
        for row in source_file:
            row = row.strip(' \n')
            if row.endswith('['):
                utterance = row.split(' ', 1)[0].strip('_')
            elif row.endswith(']'):
                row = row.strip(' ]')
                array.append([float(value) for value in row.split(' ')])
                output_file = os.path.join(output_folder, utterance + '.npy')
                np.save(output_file, np.array(array))
                n_files += 1
                array = []
            else:
                array.append([float(value) for value in row.split(' ')])
    # If an intermediary txt file was created, remove it.
    if temp_txt:
        os.remove(ark_file)
    print("Succesfully extracted %s utterances' data to .npy files." % n_files)
