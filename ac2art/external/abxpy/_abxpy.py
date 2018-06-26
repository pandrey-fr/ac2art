# coding: utf-8

"""Set of generic functions to call ABXpy module scripts.

ABXpy is a Python 2.7 package to conduct ABX discrimination tasks,
developped by the Bootphon team and distributed under MIT license
at https://github.com/bootphon/ABXpy.
"""

import os
import time


from ac2art.utils import (
    check_batch_type, check_positive_int, check_type_validity, CONSTANTS
)


ABXPY_FOLDER = CONSTANTS['abxpy_folder']


# Short but clear parameter names; pylint: disable=invalid-name
def abxpy_task(item_file, output, on, across=None, by=None):
    """Run the ABXpy task module.

    item_file : path to a database .item file used to form ABX triplets
    output    : path to the output file to write
    on        : item file column whose items to discriminate ; A and X
                will share the same `on` value, differing from that of B
    across    : optional item file column(s) ; A and B will share the same
                `across` value, differing from that of X (list or str)
    by        : optional item file column(s) ; A, B and X will share
                the same `by` value (list or str)
    """
    # Check arguments' type validity and adjust them if needed.
    check_batch_type(str, item_file=item_file, output=output, on=on)
    check_batch_type((str, list, type(None)), by=by, across=across)
    if isinstance(across, list):
        across = ' '.join(across)
    if isinstance(by, list):
        by = ' '.join(by)
    # Build the task.py call and run it.
    task = os.path.join(ABXPY_FOLDER, 'task.py') + ' -o ' + on
    if across is not None:
        task += ' -a ' + across
    if by is not None:
        task += ' -b ' + by
    status = os.system(' '.join(('python2', task, '--', item_file, output)))
    if status != 0:
        raise RuntimeError("ABXpy task.py ended with status code %s." % status)
    print('ABXpy task module was successfully run.')
# pylint: enable=invalid-name


def abxpy_distance(features_file, task_file, output, n_jobs=1):
    """Run the ABXpy distance module.

    features_file : path to a h5 file containing the features to evaluate
    task_file     : path to a task file output by the ABXpy task module
    output        : path to the output file to write
    n_jobs        : number of CPU cores to use (positive int, default 1)
    """
    check_batch_type(
        str, features_file=features_file, task_file=task_file, output=output
    )
    check_positive_int(n_jobs, 'n_jobs')
    distance = os.path.join(ABXPY_FOLDER, 'distance.py')
    distance += ' -n 1 -j %s' % n_jobs
    cmd = ' '.join(('python2', distance, features_file, task_file, output))
    status = os.system(cmd)
    if status != 0:
        raise RuntimeError(
            "ABXpy distance.py ended with status code %s." % status
        )
    print('ABXpy distance module was successfully run.')


def abxpy_score(distance_file, task_file, output):
    """Run the ABXpy score module.

    distance_file : path to a file output by the ABXpy distance module
    task_file     : path to a task file output by the ABXpy task module
    output        : path to the output file to write
    """
    check_batch_type(
        str, distance_file=distance_file, task_file=task_file, output=output
    )
    score = os.path.join(ABXPY_FOLDER, 'score.py')
    cmd = ' '.join(('python2', score, task_file, distance_file, output))
    status = os.system(cmd)
    if status != 0:
        raise RuntimeError("ABXpy score.py returned status code %s." % status)
    print('ABXpy score module was successfully run.')


def abxpy_analyze(score_file, task_file, output):
    """Run the ABXpy analyze module.

    score_file : path to a score file output by the ABXpy score module
    task_file  : path to a task file output by the ABXpy task module
    output     : path to the output file to write
    """
    check_batch_type(
        str, score_file=score_file, task_file=task_file, output=output
    )
    analyze = os.path.join(ABXPY_FOLDER, 'analyze.py')
    cmd = ' '.join(('python2', analyze, score_file, task_file, output))
    status = os.system(cmd)
    if status != 0:
        raise RuntimeError(
            "ABXpy analyze.py ended with status code %s." % status
        )
    print('ABXpy analyze module was successfully run.')


def abxpy_pipeline(features_file, task_file, output, n_jobs=1):
    """Run the ABXpy pipeline on a set of features.

    The pipeline run consists of the distance, score and analyze modules
    of ABXpy. Intermediary files will be removed, so that this function
    solely returns a .csv file summing up computed scores.

    features_file : path to a h5 file containing the features to evaluate
    task_file     : path to a task file output by the ABXpy task module
    output        : path to the output file to write
    n_jobs        : number of CPU cores to use (positive int, default 1)
    """
    check_type_validity(output, str, 'output')
    # Assign names to intermediary files.
    distance_file = '%i.distance' % time.time()
    score_file = '%i.score' % time.time()
    # Run the ABXpy pipeline.
    abxpy_distance(features_file, task_file, distance_file, n_jobs)
    abxpy_score(distance_file, task_file, score_file)
    abxpy_analyze(score_file, task_file, output)
    # Remove intermediary files.
    os.remove(distance_file)
    os.remove(score_file)
    print("Done running ABXpy. Results were written to '%s'." % output)
