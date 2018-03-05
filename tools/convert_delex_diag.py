'''
Convert raw dialog file to delexicalized file
'''

import json
from copy import deepcopy

def convert(diag_fn, delex_fn, output_fn):
    with open(diag_fn) as f:
        diags = json.load(f)

    with open(delex_fn) as f:
        delexed = f.readlines()

    delex_iter = iter(delexed)

    for diag_idx, diag in enumerate(diags):
        for turn_idx, turn in enumerate(diag["dialogue"]):
            diags[diag_idx]["dialogue"][turn_idx]["system_transcript"] = next(delex_iter).replace("\t", "").strip()

    with open(output_fn, 'w') as f:
        json.dump(diags, f, indent=4)

if __name__ == '__main__':
    convert("data/woz2_dev.json", "data/delex_dev.txt", "data/woz2_dev.delexed.json")