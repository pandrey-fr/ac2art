# coding: utf-8
#
# Copyright 2018 Paul Andrey
#
# This file is part of ac2art.
#
# ac2art is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ac2art is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ac2art.  If not, see <http://www.gnu.org/licenses/>.

"""Set functions to read and reformat some data files used with abkhazia."""

import os

import numpy as np

from ac2art.utils import check_type_validity, CONSTANTS


def copy_feats(input_file, output_file):
    """Copy data from a (set of) ark file(s) to a txt file.

    input_file  : path to a .ark, ark-like .txt file or to a .scp
                  file indexing data from one or more .ark files
                  whose data to copy
    output_file : path to a .ark, ark-like.txt file or .scp file
                  to create (.scp file are doubled with a .ark one)
    """
    # Check the arguments' validity.
    check_type_validity(input_file, str, 'input_file')
    check_type_validity(output_file, str, 'output_file')
    if not os.path.exists(input_file):
        raise FileNotFoundError("'%s' does not exist." % input_file)
    # Set up the input file's handling descriptor.
    in_ext = input_file.rsplit('.', 1)[-1]
    if in_ext == 'txt':
        input_file = 't,ark:' + input_file
    elif in_ext in ('ark', 'scp'):
        input_file = input_file[-3:] + ':' + input_file
    else:
        raise TypeError(
            'Invalid input file extension: should be ark, txt or scp.'
        )
    # Set up the output file's handling descriptor.
    out_ext = output_file.rsplit('.', 1)[-1]
    if out_ext == 'ark':
        output_file = 'ark:' + output_file
    elif out_ext == 'scp':
        output_file = 'ark,scp:{0}.ark,{0}.scp'.format(output_file[:-4])
    elif out_ext == 'txt':
        output_file = 't,ark:' + output_file
    else:
        raise TypeError(
            'Invalid output file extension: should be ark, txt or scp.'
        )
    # Set up the copy-feats command.
    command = os.path.join(CONSTANTS['kaldi_folder'], 'src/featbin/copy-feats')
    command += ' %s %s' % (input_file, output_file)
    # Run ark-to-txt conversion.
    status = os.system(command)
    if status != 0:
        raise RuntimeError(
            'kaldi copy-feats (%s to %s) exited with error code %s.'
            % (in_ext, out_ext, status)
        )
    print(
        'Successfully ran %s to %s conversion with kaldi.' % (in_ext, out_ext)
    )


def update_scp(scp_file, folder=''):
    """Update all paths in a .scp indexing file.

    scp_file : path to the scp file to alter (str)
    folder   : path to the folder containing the input files, either
               relative or absolute (str, default '', i.e. folder
               from which kaldi commands targetting the scp file are
               to be called)
    """
    check_type_validity(folder, str, 'folder')
    os.system('mv %s %s' % (scp_file, scp_file + '.tmp'))
    with open(scp_file + '.tmp', 'r') as infile:
        with open(scp_file, 'w') as outfile:
            for row in infile:
                utterance, path = row.split(' ')
                path = os.path.join(folder, os.path.basename(path))
                outfile.write(utterance + ' ' + path)
    os.system('rm %s.tmp' % scp_file)


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
    # Iteratively extract utterances' data and write individual npy files.
    n_files = 0
    for utterance, array in read_ark_file(ark_file):
        output_file = os.path.join(output_folder, utterance + '.npy')
        np.save(output_file, array)
        n_files += 1
    print("Succesfully extracted %s utterances' data to .npy files." % n_files)


def read_ark_file(filename):
    """Yield the contents of an ark, scp or ark-like txt file.

    Yield tuples consisting, for each utterance, of
    its name (str) and its values (numpy.ndarray).

    When given an ark or scp file as input, a temporary
    txt copy is first created, and later removed.
    """
    extension = filename.rsplit('.', 1)[1]
    if extension not in ('ark', 'scp', 'txt'):
        raise TypeError("'%s' is not an ark, scp or txt file." % filename)
    # Copy the records to a txt file, if relevant.
    convert = (extension != 'txt')
    if convert:
        txt_filename = filename[:-3] + 'txt'
        copy_feats(filename, txt_filename)
        filename = txt_filename
    # Read and yield records from the txt file.
    for record in _read_txt_ark_file(filename):
        yield record
    # Remove the temporary txt file, if any.
    if convert:
        os.remove(filename)


def _read_txt_ark_file(filename):
    """Yield the contents of an ark-like txt file.

    Yield tuples consisting, for each utterance, of
    its name (str) and its values (numpy.ndarray).
    """
    with open(filename) as ark_file:
        utterance = None
        array = []
        for row in ark_file:
            row = row.strip(' \n')
            if row.endswith('['):
                utterance = row.split(' ', 1)[0].strip('_')
            elif row.endswith(']'):
                row = row.strip(' ]')
                array.append([float(value) for value in row.split(' ')])
                yield (utterance, np.array(array))
                array = []
            else:
                array.append([float(value) for value in row.split(' ')])
