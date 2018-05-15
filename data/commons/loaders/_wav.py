# coding: utf-8

"""Set of tools to handle acoustic waveform data."""

import os

import numpy as np
import scipy.linalg
import scipy.signal
import librosa

from data.commons.enhance import add_dynamic_features
from utils import check_type_validity


class Wav:
    """Class to load and transform acoustic data from .wav files.

    This class loads data from a .wav file and build frames of samples
    out of it at initialisation.

    It may then produce the following representations of the audio, by frame:
      - MFCC (Mel Frequency Cepstral Coefficients), using 'get_mffc'.
      - LPC (Linear Predictive Coding) coefficients and error, using 'get_lpc'.
      - LSF (Line Spectral Frequencies) coefficients, using 'get_lsf'.
      - Root mean square energy, using 'get_rms_energy'.
    """

    def __init__(
            self, filename, sampling_rate=16000, frame_time=25, hop_time=10
        ):
        """Load the .wav data and reframe it.

        filename    : path to the .wav audio file.
        sample_rate : sampling rate of the signal, in Hz; resampling
                      will be used if needed (int, default 16000)
        frame_time  : frames duration, in milliseconds (int, default 25)
        hop_time    : number of milliseconds between each frame's
                      start time (int, default 10)
        """
        check_type_validity(filename, str, 'filename')
        self.filename = os.path.abspath(filename)
        data, sampling_rate = librosa.load(self.filename, sr=sampling_rate)
        self.signal = data
        self.sampling_rate = sampling_rate
        self.frame_length = int((frame_time * sampling_rate) / 1000)
        self.hop_length = int((hop_time * sampling_rate) / 1000)

    def __len__(self):
        """Return the number of samples."""
        return len(self.signal)

    @property
    def duration(self):
        """Return the waveform's duration in seconds."""
        return len(self) / self.sampling_rate

    def get_mfcc(self, n_coeff=13, static_only=False):
        """Return Mel-frequency cepstral coefficients for each audio frame.

        n_coeff     : number of MFCC to return for each frame
                      (positive int, default 13, maximum 40)
        static_only : whether to return the sole static MFCC features
                      instead of computing the delta and deltadelta
                      features (bool, default False)

        This implementation is based on that of 'librosa.features.mfcc',
        which it adapts so as to pass some specific options when building
        the initial spectrogram.
        """
        # Compute MFCC coefficients.
        mfcc = librosa.feature.mfcc(
            self.signal, sr=self.sampling_rate, n_mfcc=n_coeff,
            n_fft=self.frame_length, hop_length=self.hop_length
        ).T
        # Optionally return the sole static mfcc coefficients.
        if static_only:
            return mfcc
        # Otherwise, add the energy and the dynamic features to the return.
        return add_dynamic_features(mfcc)

    def get_rms_energy(self):
        """Return root mean squared energy for each audio frame."""
        return librosa.feature.rmse(
            self.signal, frame_length=self.frame_length,
            hop_length=self.hop_length
        ).T

    def get_lpc(self, n_coeff=20, static_only=False):
        """Return linear predictive coding coefficients for each audio frame.

        n_coeff     : number of LPC coefficients to return for each frame
                      (positive int, default 20)
        static_only : whether to return the sole static LPC features
                      instead of adding up the energy and compute the
                      delta and deltadelta LPC and energy features
                      (bool, default False)
        """
        frames = librosa.util.frame(
            self.signal, self.frame_length, self.hop_length
        )
        lpc, _ = linear_predictive_coding(frames, n_coeff)
        if static_only:
            return lpc
        energy = self.get_rms_energy()
        length = min(len(energy), len(lpc))
        return add_dynamic_features(
            np.concatenate([lpc[:length], energy[:length]], axis=1)
        )

    def get_lsf(self, n_coeff=20, static_only=False):
        """Return line spectral frequencies coefficients for each audio frame.

        n_coeff     : number of LPC coefficients to return for each frame
                      (positive int, default 20)
        static_only : whether to return the sole static LSF features
                      instead of adding up the energy and compute the
                      delta and deltadelta LSF and energy features
                      (bool, default False)
        """
        # Compute n + 1 LPC coefficients and convert them to LSF.
        lpc = self.get_lpc(n_coeff + 1, static_only=True)
        lsf = lpc_to_lsf(lpc)
        # Replace NaN values with zero.
        lsf[np.isnan(lsf)] = 0
        # Optionally add energy and dynamic features before returning.
        if static_only:
            return lsf
        energy = self.get_rms_energy()
        length = min(len(energy), len(lsf))
        return add_dynamic_features(
            np.concatenate([lsf[:length], energy[:length]], axis=1)
        )


def linear_predictive_coding(frames, n_coeff=20):
    """Return linear predictive coding coefficients for each audio frame.

    frames  : 2-D numpy.ndarray where each line represents an audio frame
    n_coeff : number of LPC coefficients to generate (also equal to the
              maximum autocorrelation lag order considered)
    """
    # Check arguments validity. Adjust the number of coefficients.
    check_type_validity(frames, np.ndarray, 'frames')
    if frames.ndim != 2:
        raise ValueError('`frames` should be a 2-D np.array.')
    check_type_validity(n_coeff, int, 'n_coeff')
    if n_coeff < 1:
        raise ValueError('`n_coeff` should be a strictly positive int.')
    n_coeff = min(n_coeff, frames.shape[1] - 1)
    # Compute the frame-wise LPC coefficients.
    autocorrelations = librosa.autocorrelate(frames, n_coeff + 1)
    lpc = np.array([
        # Levinson-Durbin recursion. False positive pylint: disable=no-member
        scipy.linalg.solve_toeplitz(autocorr[:-1], autocorr[1:])
        for autocorr in autocorrelations
    ])
    # Compute the frame_wise root mean squared prediction errors.
    frame_wise_errors = np.array([
        frames[:, i] - np.sum(lpc * frames[:, i - n_coeff:i][:, ::-1], axis=1)
        for i in range(n_coeff, frames.shape[1])
    ])
    frames_rmse = np.sqrt(np.mean(np.square(frame_wise_errors), axis=0))
    # Return the LPC coefficients and error terms.
    return lpc, frames_rmse


def lpc_to_lsf(lpc_coefficients):
    """Turn a numpy.ndarray of LPC coefficients to LSF coefficients.

    This implementation is largely based on that of MATLAB,
    save for the fact that it operates on LPC coefficients
    of multiple frames at the same time.

    The returned LSF coefficients have values in [0, pi].
    """
    # Normalize the LPC coefficients and add a final zero coefficient.
    lpc = lpc_coefficients.copy()
    lpc /= lpc[:, [0]]
    lpc = np.concatenate([lpc, np.zeros((len(lpc), 1))], axis=1)
    # Form the P and Q polynomials by sum and difference filters.
    p_polynoms = lpc - lpc[:, ::-1]
    q_polynoms = lpc + lpc[:, ::-1]
    # If the LPC order is odd, remove the known roots of P.
    if lpc.shape[1] % 2:
        p_polynoms = np.array([
            scipy.signal.deconvolve(polynom, np.array([1, 0, -1]))[0]
            for polynom in p_polynoms
        ])
    # If the LPC order is even, remove the known roots of P and Q.
    else:
        p_polynoms = np.array([
            scipy.signal.deconvolve(polynom, np.array([1, -1]))[0]
            for polynom in p_polynoms
        ])
        q_polynoms = np.array([
            scipy.signal.deconvolve(polynom, np.array([1, 1]))[0]
            for polynom in q_polynoms
        ])
    # Compute polynomial roots.
    p_roots = np.array([np.roots(polynom) for polynom in p_polynoms])
    q_roots = np.array([np.roots(polynom) for polynom in q_polynoms])
    # Compute roots' angle on the unit circle. Ommit complex conjugates.
    lsf = np.abs(np.concatenate(
        [np.angle(p_roots[:, ::2]), np.angle(q_roots[:, ::2])], axis=1
    ))
    # Sort and return the computed LSF coefficients.
    lsf.sort()
    return lsf


def roots_to_polynoms(roots):
    """Convert an array of roots to polynomial coefficients."""
    roots = np.concatenate([roots, np.conjugate(roots)], axis=1)
    return np.array([
        np.poly(polynom_roots) for polynom_roots in roots
    ])


def lsf_to_lpc(lsf_coefficients):
    """Turn a numpy.ndarray of LSF coefficients to LPC coefficients.

    The input lsf_coefficients are expected to have values in [0, pi].
    """
    # Restore polynomials based on the LSF coefficients.
    roots = np.exp(lsf_coefficients * complex(0, 1))
    p_polynoms = roots_to_polynoms(roots[:, 1::2])
    q_polynoms = roots_to_polynoms(roots[:, ::2])
    # Restore omitted polynomial roots based on LSF order parity.
    if lsf_coefficients.shape[1] % 2:
        p_polynoms = np.array([
            scipy.signal.convolve(polynom, np.array([1, 0, -1]))
            for polynom in p_polynoms
        ])
    else:
        p_polynoms = np.array([
            scipy.signal.convolve(polynom, np.array([1, -1]))
            for polynom in p_polynoms
        ])
        q_polynoms = np.array([
            scipy.signal.convolve(polynom, np.array([1, 1]))
            for polynom in q_polynoms
        ])
    # Add up the computed polynomials to deduce the LPC coefficients.
    lpc = (p_polynoms + q_polynoms) / 2
    return lpc[:, :-1]
