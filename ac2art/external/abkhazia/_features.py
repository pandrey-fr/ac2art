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

"""Set of functions to produce features using abkhazia."""

import os


def compute_mfcc(
        data_folder, n_coeff=13, pitch=True,
        frame_time=25, hop_time=10, output_folder=None
    ):
    """Run MFCC computation on an abkhazia corpus.

    data_folder   : path to the corpus's 'data/' folder
    n_coeff       : number of mfcc coefficients to compute (int, default 13)
    pitch         : whether to compute pitch features (bool, default True)
    frame_time    : frames duration, in milliseconds (int, default 25)
    hop_time      : frames' shift duration, in milliseconds (int, default 10)
    output_folder : optional output folder - existing folders will be
                    overwritten (default: 'mfcc_features/' folder parallel
                    to the 'data/' one)

    MFCC will be computed for each utterance in the corpus,
    including delta and deltadelta features. The first cepstral
    coefficient will be kept (i.e. not replaced by energy).

    The computed features will be stored in speaker-wise .ark
    files, indexed by a single 'feats.scp' file.
    """
    #
    if output_folder is None:
        output_folder = os.path.join(
            os.path.abspath(data_folder), '..', 'mfcc_features'
        )
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # Set up the abkhazia mfcc computation command.
    command = 'abkhazia features mfcc -v --force'
    command += ' --num-ceps %i' % n_coeff
    command += ' --delta-order 2' + ' --pitch' * pitch
    command += ' --frame-length %i --frame-shift %i' % (frame_time, hop_time)
    command += ' -o %s %s/..' % (output_folder, data_folder)
    # Run the mfcc computation.
    status = os.system(command)
    if status != 0:
        raise RuntimeError(
            'abkhazia features mfcc call exited with error code %s.' % status
        )
    print('Successfully ran mfcc computation with abkhazia.')
