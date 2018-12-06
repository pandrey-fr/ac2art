## ac2art: acoustic-to-articulatory inversion using neural networks

`ac2art` is a Python package for the supervised learning of the
acoustic-to-articulatory inversion task using neural networks. It integrates a
data processing pipeline that already interfaces some reference _corpora_ and
is ready to be expanded to additional ones, as well as a neural networks API
to build, train and save models of various and modular architectures, that
relies in the background on the low-level API of `Tensorflow`.

### Documentation

Beside the systematic documentation of all implemented functions in the form
of docstrings, basic knowledge of how `ac2art` should be used and is structured
is provided in the form of a hands-on tutorial (ipython notebook with an html
counterpart) and three markdown files covering respectively the `ac2art.corpora`
module, the `ac2art.networks` one and the `ac2art.networks.NeuralNetwork` API.
Those files are to be found in the `docs` folder of this repository.

### Installation

#### 1. Software requisites

**Python**

A Python 3 (>= 3.4) installation is required to run `ac2art`.

Additionally, a Python 2.7 installation is required as some key functionalities
rely on interfaced third-party Python 2 packages (see below).

**Python 2 third-party packages**

The (optional) use of [kaldi](http://kaldi-asr.org/) to compute acoustic
features depends on the [abkhazia](https://github.com/bootphon/abkhazia)
package. The computation of ABX discriminability metrics depends on
[ABXpy](https://github.com/bootphon/ABXpy). Both those packages should be
installed manually before installing `ac2art` (and will be checked for at
installation time). Please **do not** use `pip` to install those packages,
are they are not maintained up-to-date on Pypy and some key functionalities
would therefore be missing and cause crashes.


**Ptyhon 3 third-party packages**

`ac2art` also depends on the following third-party Python 3 packages, which
will be automately installed as part of the installation procedure :
`h5features`, `numpy`, `pandas`, `scipy`, `sphfile` and `tensorflow`.
You may want to manually compile and install the latter depending on your
system. Note that as of now, `ac2art` does not take advantage of GPUs.

#### 2. Data _corpora_

To this day, `ac2art` supports the following data _corpora_ :
[MNGU0](http://www.mngu0.org/),
[MOCHA-TIMIT](http://www.cstr.ed.ac.uk/research/projects/artic/mocha.html) and
[MSPKA](http://www.mspkacorpus.it/) (with some to-be-fixed flaws for the latter).
One may choose not to download them prior to installing the package ; however,
doing so enables the inclusion of their path to the `config.json` file (see
following section).

So as to work with the data from the _corpora_, please do not rename nor
restructure the downloaded files and folders, and simply copy them to any
location that you see fit (and will then have to fill in the `config.json`
file).

For the MNGU0 and MOCHA-TIMIT packages, acoustic features computation using
Kaldi requires the use of completed phoneme labeling files, which were manually
filled based on those distributed with each _corpus_ and are yet to be released.
In the meanwhile, one may use an `ac2art` built-in feature to compute MFCC,
resulting in slightly different coefficients and not including the pitch
features which Kaldi adds.

#### 3. Configuration file

`ac2art` depends on a `config.json` file that should be created and set up
before running its installation, which will be copied somewhere on your system
during the installation. It may be updated later using the built-in
`ac2art.utils.update_config` function, which may however require you to run
python with root rights depending on the way you installed the package.

An `example_config.json` file is provided as part of this repository, which
can be renamed to `config.json` and filled out, as it contains the fields that
you should want to use. Those include the mandatory paths to your kaldi and
ABXpy installations, as well as the (optional but fairly crucial) paths to the
folders containing the raw data from the supported _corpora_, and paths to
folders where to store the processed versions of those datasets (including
acoustic and articulatory features in both raw and normalized versions, as
well as produced ABXpy h5 features files and output scores).

#### 4. Install the package

To install `ac2art` on your machine:
1. Download a copy of the Git repository:
`git clone https://github.com/pandrey-fr/ac2art.git`
2. Set up the `config.json` file in the downloaded folder (see previous section).
3. In the command line, `cd` to the folder, then run `python3 setup.py install`
(either with root rights, or adding the `--user` option for local installation).

### License

**Copyright 2018 Paul Andrey**

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.

### Contact

If you run into any issue regarding `ac2art`, you may open an issue on this
Github repository, or contact [the author](https://github.com/pandrey-fr).
