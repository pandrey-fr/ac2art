# coding: utf-8

"""Wrapper functions to build raw data loaders."""


def build_utterances_getter(get_speaker_utterances, speakers, corpus):
    """Build a function returning lists of utterances of a corpus."""
    def get_utterances_list(speaker=None):
        """Return the list of {0} utterances from a given speaker."""
        nonlocal speakers, get_speaker_utterances
        if speaker is None:
            return [
                utterance for speaker in speakers
                for utterance in get_speaker_utterances(speaker)
            ]
        elif speaker in speakers:
            return get_speaker_utterances(speaker)
        raise KeyError("Invalid speaker: '%s'." % speaker)
    # Adjust the function's docstring and return it.
    get_utterances_list.__doc__ = get_utterances_list.__doc__.format(corpus)
    return get_utterances_list
