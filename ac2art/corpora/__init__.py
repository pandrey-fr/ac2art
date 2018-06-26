# coding: utf-8

"""Submodule defining corpus-specific data processing pipelines.


As of now, the supported corpora include the following:
  mngu0 : mngu0 day-one acoustic and ema data
  mocha : mocha-timit data for the two initial speakers
  mspka : mspka first session data


For each supported corpus, a `ac2art.corpora.<corpus>` submodule
is defined, which should be explicitely imported by the user (to
avoid useless overhead, these submodules are not processed upon
importing ac2art).


Note that using a corpus requires to actually dispose of the data,
and to have referred their location, as well as that the folder in
which pre-processed data is to be written, in the config.json file
at the root of the package.


Each corpus's submodule consists in multiple units of code, which
respect the following mandatory structure of implemented functions
(and may add some internal dependency functions):


`ac2art.corpora.<corpus>.abkhazia`
`ac2art.corpora.<corpus>.raw`
`ac2art.corpora.<corpus>.preprocess`
`ac2art.corpora.<corpus>.load`
`ac2art.corpora.<corpus>.abx`


`ac2art.corpora.prototype` mirrors the last four points of this
structure, and adds a `utils` submodule made of locally-relevant
tools, such as some generic data loaders. It otherwise implements
functions-defining functions which may be used to create the large
majority of the mandatory functions for each corpus, ensuring that
they behave accordingly to both the defined data processing API and
the corpus's specificities.


To add support for a corpus, one should thus implement the necessary
functions in the `raw` and `abkhazia` subunits, then use the prototype
wrappers to build the rest of the subunits and functions, following
what is done for the already supported corpora.


Note that for the mocha and mspka corpora, adding support to
data from additional recording sessions / speakers should be
straight-forward, requiring only to edit the list of speakers
in `ac2art.corpora.<corpus>.raw._loaders.py` (SPEAKERS) constant.
""" # FIXME: complete and pass to a .md file, then load it here


from . import prototype
