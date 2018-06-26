# coding: utf-8

"""Set of generic functions to call ABXpy module scripts.

ABXpy is a Python 2.7 package to conduct ABX discrimination tasks,
developped by the Bootphon team and distributed under MIT license
at https://github.com/bootphon/ABXpy.

Running functions from this submodule requires to have downloaded
a copy of the ABXpy Git repository, installed its dependencies for
python 2.7 and referenced its path under the 'abxpy_folder' key in
this package's 'config.json' file.

At the time of the latest revision of this submodule, the version
of ABXpy was 0.1.0, and the latest commit to the ABXpy repository
was that of SHA 0fe520e09d1bf3f580706f67412890815f2f3c93.
"""

from ._abxpy import (
    abxpy_pipeline, abxpy_task, abxpy_distance, abxpy_score, abxpy_analyze
)
