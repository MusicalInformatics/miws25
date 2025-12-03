#!/usr/bin/env python
import pickle
import bz2


def load_pyc_bz(fn):
    return pickle.load(bz2.BZ2File(fn, "r"))


def save_pyc_bz(d, fn):
    pickle.dump(d, bz2.BZ2File(fn, "w"), pickle.HIGHEST_PROTOCOL)
