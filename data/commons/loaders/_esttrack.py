# coding: utf-8

"""Class to handle EST Track files used in the Mocha-Timit and mngu0 corpora."""

import numpy as np

from data.commons.loaders import AbstractDataLoader


class EstTrack(AbstractDataLoader):
    """Class to load data from EST Track files.

    EST Track files include .ema and .lsf files, respectively recording
    EMA articulatory data and pre-processed LSF representations of acoustic
    data.
    """

    def __init__(self, filename):
        """Initialize the EstTrack instance.

        filename : path to the Est Track file (str)
        """
        self.name = None
        super().__init__(filename)

    def load(self):
        """Load the Est Track data from file."""
        # Read and parse the file.
        with open(self.filename, 'rb') as infile:
            # Check file validity.
            if next(infile).decode('latin-1').strip('\n') != 'EST_File Track':
                raise ValueError(
                    "'%s' is not an EST Track file." % self.filename
                )
            # Parse the file's header for meta-information.
            n_frames, n_columns, storage_type, byte_order = (
                self._parse_header(infile)
            )
            # Load the actual data.
            if storage_type == 'binary':
                data = np.fromfile(infile, np.dtype(byte_order + 'f'))
                expected_shape = (n_frames * (n_columns + 2),)
                if data.shape != expected_shape:
                    raise ValueError(
                        'Dimension error: expected %s values, got %s.'
                        % (expected_shape, data.shape)
                    )
                data = data.reshape(n_frames, n_columns + 2)
            elif storage_type == 'ascii':
                data = np.genfromtxt(infile)
            else:
                raise ValueError(
                    "Unexpected data storage type: '%s'." % storage_type
                )
        # Assign the loaded data to the instance's attributes.
        self.time_index = data[:, 0]
        self.data = data[:, 2:]

    def _parse_header(self, infile):
        """Read the Est Track file's header and parse information out of it.

        infile : an open stream to the file, read in binary mode.

        Return the number of data frames (int) and columns (int), and
        strings indicating the data storage type and its byte order.
        """
        storage_type = 'binary'
        byte_order = 'l'
        for line in infile:
            line = line.decode('latin-1').strip('\n')
            if line == 'EST_Header_End':
                break
            elif line.startswith('DataType'):
                storage_type = line.rsplit(' ', 1)[1]
            elif line.startswith('ByteOrder'):
                byte_order = {'01': '<', '10': '>'}[line[-2:]]
            elif line.startswith('NumFrames'):
                n_frames = int(line.rsplit(' ', 1)[-1])
            elif line.startswith('NumChannels'):
                n_columns = int(line.rsplit(' ', 1)[-1])
            elif line.startswith('Channel_'):
                col_id, col_name = line.split(' ', 1)
                self.column_names[int(col_id.split('_', 1)[-1])] = col_name
            elif line.startswith('name'):
                self.name = line.rsplit(' ', 1)[-1]
        return n_frames, n_columns, storage_type, byte_order
