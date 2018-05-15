# coding: utf-8

"""Generic functions to prepare abkhazia corpora' data/ folders."""

import os

import pandas as pd

from data.utils import CONSTANTS
from utils import check_type_validity, import_from_string


def prepare_abkhazia_corpus(
        corpus, data_folder, limit_phones=True, mode='w', id_length=None
    ):
    """Build or complete a corpus's data/ folder for use with abkhazia.

    corpus       : name of the corpus whose data to prepare (str)
    data_folder  : path to the 'data/' folder to build or complete
    limit_phones : whether to map the corpus' phones to a restricted set
                   of IPA phones, thus aggregating some (bool, default True)
    mode         : file writing mode (either 'w' or 'a', default 'w')
    id_length    : optional fixed length of utterances' id used internally

    Note: the `mode` and `id_length` parameters may be used to pile up
          data from multiple corpora in a single data/ folder, thus
          having abkhazia treat them as one large corpus. In this case,
          please be careful about corpus-specific phone symbols overlap.
    """
    # Check arguments validity.
    check_type_validity(corpus, str, 'corpus')
    check_type_validity(data_folder, str, 'data_folder')
    check_type_validity(limit_phones, bool, 'limit_phones')
    if mode not in ('w', 'a'):
        raise TypeError("'mode' should be a str in {'a', 'w'}.")
    check_type_validity(id_length, (int, type(None)), 'id_length')
    # Make the output directories if needed.
    wav_folder = os.path.join(data_folder, 'wavs')
    for folder in (data_folder, wav_folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)
    # Gather dependency functions.
    copy_wavs, get_transcription = import_from_string(
        'data.%s.abkhazia._loaders' % corpus,
        ['copy_wavs', 'get_transcription']
    )
    # Copy wav files to the data folder and gather the utterances list.
    utt_files = copy_wavs(wav_folder)
    utt_ids = normalize_utterance_ids(utt_files, id_length)
    # Fill the segments.txt file.
    with open(os.path.join(data_folder, 'segments.txt'), mode) as abk_file:
        abk_file.write('\n'.join(
            name + ' ' + name.strip('_') + '.wav' for name in utt_ids
        ))
    # Build the utt2spk, spk2utt, text and lexicon files.
    make_utt2spk_files(data_folder, utt_ids, mode)
    make_text_files(data_folder, utt_ids, get_transcription, mode)
    # Build the phones, silences and variants files.
    make_phones_files(data_folder, corpus, limit_phones, mode)


def normalize_utterance_ids(utterances, id_length=None):
    """Produce a list of fixed-length utterances names copies."""
    max_length = max([len(x) for x in utterances])
    if id_length is None:
        id_length = max_length
    elif id_length < max_length:
        raise ValueError(
            "'id_length' argument is too small: longest id is %s." % max_length
        )
    return [name + '_' * (id_length - len(name)) for name in utterances]



def make_utt2spk_files(data_folder, utt_ids, mode='w'):
    """Fill the utt2spk and spk2utt txt files for abkhazia."""
    # Gather speaker-wise utterances list.
    speaker_utterances = {}
    for name in utt_ids:
        speaker = name.split('_', 1)[0]
        speaker_utterances.setdefault(speaker, [])
        speaker_utterances[speaker].append(name)
    # Fill the utt2spk file.
    with open(os.path.join(data_folder, 'utt2spk.txt'), mode) as abk_file:
        for speaker, utterances in speaker_utterances.items():
            abk_file.write('\n'.join(
                utterance + ' ' + speaker for utterance in utterances
            ) + '\n')
    # Fill the spk2utt file.
    with open(os.path.join(data_folder, 'spk2utt.txt'), mode) as abk_file:
        abk_file.write('\n'.join(
            speaker + ' ' + ' '.join(utterances)
            for speaker, utterances in speaker_utterances.items()
        ))


def make_text_files(data_folder, utt_ids, get_transcription, mode='w'):
    """Fill the text and lexicon files for abkhazia."""
    # TO FIX: phones symbols overlap when dealing with multiple corpora.
    # TO DO : add an option to replace symbols with IPA ones directly.
    if mode == 'a':
        with open(os.path.join(data_folder, 'lexicon.txt')) as lex_file:
            next(lex_file)
            lexicon = {row.split(' ', 1)[0] for row in lex_file}
    else:
        lexicon = set()
    # Write the text file and build the lexicon on the go.
    with open(os.path.join(data_folder, 'text.txt'), mode) as abk_file:
        for name in utt_ids:
            transcript = get_transcription(name.strip('_'), phonetic=True)
            lexicon.update({word for word in transcript.split(' ')})
            abk_file.write(name + ' ' + transcript + '\n')
    # Order and clean the lexicon.
    lexicon = sorted(lexicon)
    if '' in lexicon:
        lexicon.remove('')
    # Write the lexicon file.
    with open(os.path.join(data_folder, 'lexicon.txt'), 'w') as abk_file:
        abk_file.write('<unk> SPN\n')
        abk_file.write('\n'.join(
            word + ' ' + word.replace('-', ' ') for word in lexicon
        ))


def make_phones_files(data_folder, corpus, limit_phones=True, mode='w'):
    """Fill the phones, silences and variants txt files for abkhazia."""
    # Fill the phones.txt file.
    symbols = pd.read_csv(CONSTANTS['%s_symbols' % corpus])
    symbols[['symbols', 'ipa' + '_reduced' * limit_phones]].to_csv(
        os.path.join(data_folder, 'phones.txt'), sep=' ',
        header=False, index=False, mode=mode
    )
    # Fill the silences.txt file.
    with open(os.path.join(data_folder, 'silences.txt'), 'w') as abk_file:
        abk_file.write('SIL\nSPN')
    # Create an empty variants.txt file.
    os.system('touch ' + os.path.join(data_folder, 'variants.txt'))
