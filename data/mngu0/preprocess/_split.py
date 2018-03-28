# coding: utf-8

"""Set of functions to split the mngu0 dataset ensuring triphones coverage"""

import os

import numpy as np

from data.mngu0.raw import load_phone_labels, get_utterances_list
from data.utils import check_type_validity, CONSTANTS


RAW_FOLDER = CONSTANTS['mngu0_raw_folder']
NEW_FOLDER = CONSTANTS['mngu0_processed_folder']


def load_triphones(name):
    """Load the set of triphones contained in a given mngu0 utterance."""
    labels = load_phone_labels(name)
    return {
        '_'.join([phone[1] for phone in labels[i:i + 3]])
        for i in range(len(labels) - 2)
    }


def build_triphones_indexer(limit):
    """Build an index of triphones appearing in each mngu0 utterance.

    limit : minimum number of utterances a triphone must appear in
            to be indexed

    Return a dict associating the (reduced) set of triphones comprised
    in a mngu0 utterance with this utterance's name.
    """
    # Gather the sets of utterances' triphones and the full list of these.
    utterances = {
        name: load_triphones(name) for name in get_utterances_list()
    }
    all_triphones = {
        triphone for utt_triphones in utterances.values()
        for triphone in utt_triphones
    }
    # Select the triphones of interest.
    triphones = {
        triphone for triphone in all_triphones
        if sum(triphone in utt for utt in utterances.values()) >= limit
    }
    # Reduce the dict referencing triphones associated to each utterance.
    utterances_index = {
        utterance: [phone for phone in utt_triphones if phone in triphones]
        for utterance, utt_triphones in utterances.items()
    }
    return utterances_index


def build_initial_split(indexer):
    """Return a split of mngu0 data that ensures good triphones coverage.

    indexer : a dict associating sets of comprised triphones to
              utterances (as returned by `build_triphones_indexer`)

    Return a tuple of three lists of utterances, representing the
    train, validation and test filesets, ensuring that all triphones
    of interest (i.e. appearing in the provided indexer) are comprised
    at least once in each of the filesets.
    """
    filesets = [{'utt': [], 'phones': set()} for _ in range(3)]
    # Iterate over the list of utterances, in random order.
    for utterance in np.random.permutation(get_utterances_list()):
        # Compute the number of covered triphones each dataset
        # would gain from incorporating the current utterance.
        gains = [
            sum(phone not in dataset['phones'] for phone in indexer[utterance])
            for dataset in filesets
        ]
        # Assign the utterance to the dataset that benefits the most of it.
        chosen = np.argmax(gains)
        filesets[chosen]['utt'].append(utterance)
        filesets[chosen]['phones'] = (
            filesets[chosen]['phones'].union(indexer[utterance])
        )
    # Check that all triphones appear in each dataset.
    all_covered = (
        filesets[0]['phones'] == filesets[1]['phones'] == filesets[2]['phones']
    )
    # If there are coverage issues, start over again.
    if not all_covered:
        print(
            'Error: invalid initial repartition. The algorithm was restarted.'
        )
        return build_initial_split(indexer)
    # Otherwise, return the designed filesets.
    return [dataset['utt'] for dataset in filesets]



def adjust_filesets(filesets, pct_train, indexer):
    """Transfer some utterances between filesets to adjust their sizes.

    filesets  : a tuple of three lists of utterances, representing the
                filesets (as returned by `build_initial_split`)
    pct_train : percentage of observations used as training data
                (float between 0 and 1) ; the rest will be divided
                equally between the validation and test sets
    indexer   : a dict associating sets of comprised triphones to
                utterances (as returned by `build_triphones_indexer`)

    Return a tuple of three lists of utterances, representing the
    train, validation and test filesets.

    The utterances moved from a fileset to another are selected
    randomly, under the condition that their removal does not
    deprive the initial fileset from a triphone of interest.
    """
    # Compute theoretical sizes of the filesets.
    total = sum(len(dataset) for dataset in filesets)
    n_obs = [0] * 3
    n_obs[0] = int(np.ceil(total * pct_train))
    n_obs[1] = int(np.floor((total - n_obs[0]) / 2))
    n_obs[2] = total - sum(n_obs)
    taken_out = []
    # Remove observations from filesets which are too large,
    # making sure that it does not alter triphones coverage.
    for dataset, size in zip(filesets, n_obs):
        if len(dataset) > size:
            n_moves = len(dataset) - size
            moved = []
            while len(moved) < n_moves:
                chosen = np.random.choice(dataset)
                movable = all(
                    sum(phone in indexer[utt] for utt in dataset) > 1
                    for phone in indexer[chosen]
                )
                if movable:
                    moved.append(chosen)
                    dataset.remove(chosen)
            taken_out.extend(moved)
    # Dispatch the selected observations in the filesets which are too small.
    for dataset, size, in zip(filesets, n_obs):
        for _ in range(size - len(dataset)):
            chosen = np.random.choice(taken_out)
            dataset.append(chosen)
            taken_out.remove(chosen)
    # Return the adjusted filesets.
    return filesets


def store_filesets(filesets):
    """Write the mngu0 filesets stored in a given dict to txt files."""
    check_type_validity(filesets, dict, 'filesets')
    # Build the output 'filesets' folder if needed.
    output_folder = os.path.join(NEW_FOLDER, 'filesets')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # Iteratively write the filesets to their own txt file.
    for set_name, fileset in filesets.items():
        path = os.path.join(output_folder, set_name + '.txt')
        with open(path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(fileset))


def split_dataset(pct_train=.7, limit=9, seed=None):
    """Split the mngu0 dataset, ensuring good triphones coverage of the sets.

    limit     : minimum number of utterances a triphone must appear in
                so as to be taken into account (int, default 9)
    pct_train : percentage of observations used as training data; the
                rest will be divided equally between the validation
                and test sets float (between 0 and 1, default .7)
    seed      : optional random seed to use

    Produce three lists of utterances, composing train, validation
    and test filesets. The filesets are built so that each triphone
    present in at list `limit` utterances appears at least once in
    each fileset. The filesets' length will also be made to match
    the `pct_train` argument.

    To achieve this, the split is conducted in two steps.
    * First, utterances are iteratively drawn in random order, and
    added to the fileset to which they add the most not-yet-covered
    triphones. This mechanically results in three filesets correctly
    covering the set of triphones (it not, the algorithm is restarted)
    whose sizes tend to approach a 70% / 30% / 30% split.
    * Then, utterances are randomly removed from the fileset(s) which
    prove too large compared to the desired split, under the condition
    that their removal does not break the triphones-coverage property.
    These utterances are then randomly re-assigned to the filesets
    which are too small.

    Note: due to the structure of the mngu0 utterances, using a `limit`
          parameter under 9 will generally fail.

    The produced filesets are stored to the filesets/ subfolder of the
    processed mngu0 folder, under the 'train_new', 'validation_new' and
    'test_new' txt files.
    """
    # Check arguments' validity
    check_type_validity(pct_train, float, 'pct_train')
    check_type_validity(limit, int, 'limit')
    if not 0 < pct_train < 1:
        raise ValueError('Invalid `pct_train` value: %s.' % pct_train)
    if limit < 3:
        raise ValueError('Minimum `limit` value is 3.')
    elif limit < 9:
        print('Warning: using such a low `limit` value is due to fail.')
    # Build the filesets.
    np.random.seed(seed)
    indexer = build_triphones_indexer(limit)
    filesets = build_initial_split(indexer)
    filesets = adjust_filesets(filesets, pct_train, indexer)
    # Write the produced filesets to txt files.
    store_filesets(dict(zip(('train', 'validation', 'test'), filesets)))


def adjust_provided_filesets():
    """Adjust the filesets lists provided with mngu0 to fit the raw data.

    As the provided filesets lists are based on a extracted version of
    the mngu0 corpus anterior to the splitting of some of the utterances
    within the raw data, running this function is necessary so as to use
    similar train, validation and test utterances of newly-processed data
    as in most papers using this database.
    """
    # Load the full list of utterances.
    utterances = get_utterances_list()
    # Iteratively produce the adjusted filesets.
    filesets = {}
    for set_name in ['train', 'validation', 'test']:
        # Load the raw fileset list.
        path = os.path.join(RAW_FOLDER, 'ema_filesets', set_name + 'files.txt')
        with open(path) as file:
            raw_fileset = [row.strip('\n') for row in file]
        # Derive the correct fileset of newly processed utterances.
        filesets[set_name + '_initial'] = [
            utt for utt in utterances if utt.strip('abcdef') in raw_fileset
        ]
    # Write the filesets to txt files.
    store_filesets(filesets)
