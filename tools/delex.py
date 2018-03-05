'''
delex.py

delexicalize system responses

'''

import re
import json
from collections import defaultdict

from nltk import wordpunct_tokenize

class Delexicalizer:
    '''
    Delexicalizer

    "Its in the center of the city. " ->
        "Its in the <v.>"
    '''

    def __init__(self, info_slots, semi_dict, values, replaces):
        '''
        info_slot = ["area", "food", ...]

        semi_dict = {
            "area" : ["area of town", ..., ""]
        }

        values = {
            "address" : ["10 x road ...", ..., ""]
        }
        '''

        self.info_slots = info_slots
        self.semi_dict = semi_dict
        self.values = values
        self.replaces = replaces # additional replaces

        self.inv_info_slots = self._inverse_dict(self.info_slots, "%s")
        self.inv_values = self._inverse_dict(self.values, "<v.%s> ", func=lambda x: x.upper())
        self.inv_semi_dict = self._inverse_dict(self.semi_dict, "%s")

        self.inv_semi_dict = { k : "<v.%s> " % self.inv_info_slots[v].upper()
                        if v in self.inv_info_slots else "<s.%s> " % v.upper() for k, v in self.inv_semi_dict.iteritems()}

        self.num_matcher = re.compile(r' \d{1,2}([., ])')
        self.post_matcher = re.compile(r'( [.]?c\.b[.]?[ ]?\d[ ]?[,]?[ ]?\d[.]?[ ]?[a-z][\.]?[ ]?[a-z][\.]?)|( cb\d\d[a-z]{2})')

        # 01333 363471

        self.phone_matcher = re.compile(r'[ (](#?0)?(\d{10}|\d{4}[ ]\d{5,6}|\d{3}-\d{3}-\d{4})[ ).,]')
        self.street_matcher = re.compile(r' (([a-z]+)?\d{1,3}([ ]?-[ ]?\d+)? )?[a-z]+ (street|road|avenue)(, (city [a-z]+))?')

    def _inverse_dict(self, d, fmt="%s ", func=str):
        inv = {}
        for k, vs in d.iteritems():
            for v in vs:
                inv[v.lower()] = fmt % (func(k))

        return inv

    def delex(self, sent):
        sent = " " + sent.lower()

        sent = self.post_matcher.sub(" <v.POSTCODE> ", sent)

        sent = " , ".join(sent.split(","))
        

        for r, v in self.replaces:
            sent = sent.replace(" " + r + " ", " " + v + " ")

        sent = sent.replace("  ", " ")

        sent = self.phone_matcher.sub(" <v.PHONE> ", sent)
        
        for v in sorted(self.inv_values.keys(), key=len, reverse=True):
            sent = sent.replace(v, self.inv_values[v])

        sent = self.street_matcher.sub(" <v.ADDRESS> ", sent)
        for v in sorted(self.inv_semi_dict.keys(), key=len, reverse=True):
            sent = sent.replace(v, self.inv_semi_dict[v])

        

        sent = self.num_matcher.sub(" <COUNT> ", sent)

        sent = sent.replace("  ", " ")

        return sent.strip()

def make_delexicalizer(semi_dict_fn, kb_fn, onto_fn, replace_fn, req_slots=["address", "phone", "postcode", "name"]):
    semi_dict = defaultdict(list)
    values = defaultdict(list)

    with open(kb_fn) as f:
        kb = json.load(f)

    with open(semi_dict_fn) as f:
        semi_dict = json.load(f)

    with open(onto_fn) as f:
        onto_data = json.load(f)

    with open(replace_fn) as f:
        replaces = [ line.strip().split('\t') for line in f.readlines() ]

    for entry in kb:
        for slot in req_slots:
            if slot in entry:
                values[slot].append(entry[slot]) 
    
    slots = ["area", "food", "pricerange", "address", "phone", "postcode", "name"]

    return Delexicalizer(onto_data["informable"], semi_dict, values, replaces)

if __name__ == '__main__':
    delex = make_delexicalizer("data/CamRestHDCSemiDict.json", "data/CamRest.json", "data/CamRestOTGY.json", "data/replace.txt")

    with open("data/woz2_dev.json") as f:
        diags = json.load(f)

    for diag in diags:
        for turn in diag["dialogue"]:
            print delex.delex(turn["system_transcript"])
    

