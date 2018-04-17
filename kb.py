import json
from functools import reduce
from collections import defaultdict

def load_kb(kb_fn, primary):
    with open(kb_fn) as f:
        data = json.load(f)

    kb = KnowledgeBase(data[0].keys(), primary)

    for obj in data:
        kb.add(obj)

    return kb

class KnowledgeBase:
    '''
    Knowledge Base

    provide api for knowledge retriving.
    '''

    def __init__(self, columns, primary):
        self.columns = columns
        self.primary = primary
        self.index = { k : defaultdict(list) for k in self.columns }
        self.objs = {}

    def add(self, obj):
        '''add a obj into this KB'''
        for k, v in obj.iteritems():
            self.index[k][v].append(obj[self.primary])
        
        self.objs[obj[self.primary]] = obj

    def get(self, primary):
        '''get object using primary key'''
        return self.objs[primary]

    def search(self, key, value):
        return self.index[key][value];

    def search_multi(self, kvs):
        '''multi-key search'''
        return reduce(lambda y, x: y & set(self.index[x[0]][x[1]]) 
                        if len(y) > 0  else y | set(self.index[x[0]][x[1]]), kvs, set())

if __name__ == '__main__':
    kb = load_kb("data/CamRest.json", "name")

    print kb.get(list(kb.search_multi([['pricerange', 'cheap'], ['area', 'east']]))[0]).get("name")