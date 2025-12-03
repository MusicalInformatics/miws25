#!/usr/bin/env python

import os
import json
import argparse
import tarfile
import io
from urllib.request import urlopen
import urllib
import re
import warnings

from IPython.display import display, HTML, Audio, update_display
import ipywidgets as widgets
import appdirs
from collections import defaultdict


TIMEOUT = 2
REPO_NAME = "vienna4x22"
DATASET_BRANCH = "master"
OWNER = "CPJKU"
DATASET_URL = "https://api.github.com/repos/{}/{}/tarball/{}".format(
    OWNER,
    REPO_NAME,
    DATASET_BRANCH,
)

# oggs will be downloaded from here
OGG_URL_BASE = "https://spocs.duckdns.org/vienna_4x22/"

TMP_DIR = appdirs.user_cache_dir("basismixer")
CFG_FILE = os.path.join(TMP_DIR, "cache.json")
CFG = None
# DATASET_DIR will be set to the path of our data
DATASET_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "vienna4x22",
    )
)
PIECES = ()
PERFORMERS = ()
SCORE_PERFORMANCE_PAIRS = None


def load_cfg():
    global CFG
    if os.path.exists(CFG_FILE):
        with open(CFG_FILE) as f:
            CFG = json.load(f)
    else:
        CFG = {"last_dataset_dir": None}


def save_cfg():
    with open(CFG_FILE, "w") as f:
        json.dump(CFG, f)


def pair_files(
    folder_dict: dict,
    full_path: bool = True,
    by_prefix: bool = True,
) -> dict:
    """Pair files in different directories by their filenames.

    The function returns a dictionary where the keys are the matched
    part of the filenames and the values are dictonaries. The keys of
    each of these dictionaries coincide with the keys in
    `folder_dict`. The value for a given key is a set of paired files
    in the corresponding folder.

    Parameters
    ----------
    folder_dict : dict
        Dictionary with arbitrary labels as keys and directory paths
        as values.
    full_path : bool, optional
        When True, return the full paths of the files. Otherwise only
        filenames are returned, omitting the directories.Defaults to
        True.
    by_prefix : bool, optional
        When True two files in different directories are paired
        whenever one filename (excluding the extension) is a prefix of
        the other. Otherwise files are only paired when the filenames
        excluding the extensions are equal. Defaults to True.

    Returns
    -------
    dict
        A dictionary with the paired files..

    """
    result = defaultdict(lambda: defaultdict(set))

    for label, directory in folder_dict.items():
        for f in os.listdir(directory):
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                name = os.path.splitext(f)[0]
                if full_path:
                    result[name][label].add(path)
                else:
                    result[name][label].add(f)

    if by_prefix and result:

        # sort by length
        snames = sorted(result.keys(), key=lambda x: len(x))
        # sort lexicographically
        snames.sort()
        cur = snames.pop(0)
        merged = set()
        while snames:
            nxt = snames.pop(0)
            if nxt.startswith(cur):
                for k, v in result[nxt].items():
                    if k in result[cur]:
                        result[cur][k].update(v)
                    else:
                        result[cur][k] = v
                merged.add(nxt)
            else:
                cur = nxt

        for n in merged:
            del result[n]

    # remove_incomplete items
    labels = set(folder_dict.keys())
    todo_delete = [k for k, k_labels in result.items() if not set(k_labels) == labels]
    for k in todo_delete:
        del result[k]

    return result


def init_dataset():
    global DATASET_DIR, PIECES, PERFORMERS, SCORE_PERFORMANCE_PAIRS

    load_cfg()

    status = widgets.Output()
    display(status)
    status.clear_output()


    if DATASET_DIR is None:
        status.append_stdout("No internet connection?\n")

    elif os.path.exists(DATASET_DIR):

        status.append_stdout("Vienna 4x22 Corpus already downloaded.\n")
        status.append_stdout("Data is in {}".format(DATASET_DIR))

    else:
        status.append_stdout("Downloading Vienna 4x22 Corpus...")
        try:
            try:
                urldata = urlopen(DATASET_URL).read()
            except urllib.error.URLError as e:
                # warnings.warn('{} (url: {})'.format(e, DATASET_URL))
                status.append_stdout("error. No internet connection?\n")
                return

            with tarfile.open(fileobj=io.BytesIO(urldata)) as archive:
                folder = next(iter(archive.getnames()), None)
                archive.extractall(TMP_DIR)
                if folder:
                    DATASET_DIR = os.path.join(TMP_DIR, folder)
                    CFG["last_dataset_dir"] = DATASET_DIR
                    save_cfg()
                # assert DATASET_DIR == os.path.join(TMP_DIR, folder)

        except Exception as e:
            status.append_stdout("\nError: {}".format(e))
            return None
        status.append_stdout("done\nData is in {}".format(DATASET_DIR))

    if DATASET_DIR is None:
        return None

    folders = dict(
        musicxml=os.path.join(DATASET_DIR, "musicxml"),
        match=os.path.join(DATASET_DIR, "match"),
    )

    SCORE_PERFORMANCE_PAIRS = []
    paired_files = pair_files(folders)
    pieces = sorted(paired_files.keys())
    for piece in pieces:
        xml_fn = paired_files[piece]["musicxml"].pop()
        for match_fn in sorted(paired_files[piece]["match"]):
            SCORE_PERFORMANCE_PAIRS.append((xml_fn, match_fn))

    fn_pat = re.compile("(.*)_(p[0-9][0-9])\.match")
    match_files = os.listdir(os.path.join(DATASET_DIR, "match"))
    pieces, performers = zip(
        *[m.groups() for m in [fn_pat.match(fn) for fn in match_files] if m]
    )
    PIECES = sorted(set(pieces))
    PERFORMERS = sorted(set(performers))


if __name__ == "__main__":
    init_dataset()
