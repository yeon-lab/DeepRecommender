import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
from utils.base import load_data, transform_category, sample, data_split

SEED = 123
np.random.seed(SEED)

def preprocessing(path, config):
    # rating info
    ratings = load_data(os.path.join(path,'u.data'),'\t')
    ratings = pd.DataFrame(ratings, columns=['UserID','MovieID','Rating','timestamp']).dropna(subset=('UserID','MovieID','Rating')).iloc[:,:-1]
    ratings = ratings[ratings['MovieID']!=267]
    
    # user info
    # categories
    users     = load_data(os.path.join(path,'u.user'),'|')
    users     = pd.DataFrame(users, columns=['UserID','Age','Gender','Occupation','Zip-code']).dropna()
    u_categories = users.copy()
    u_categories = u_categories.iloc[:,1:]
    u_categories, user_count= transform_category(u_categories)
    u_categories['UserID'] = users['UserID']

    # item info
    movies = load_data(os.path.join('u.item'),'|')
    items = []  
    for movie in movies[:-1]:
        items.append([movie[0],
           movie[1][:len(movie[1])-6],movie[2][-4:],
           movie[5:]
            ])
     
    items = pd.DataFrame(items, columns=['MovieID','Title','year','Genres']).dropna()
    
    # category   
    it_categories = items.copy()
    it_categories = it_categories[['year']]
    it_categories, item_count = transform_category(it_categories)
    it_categories['MovieID'] = items['MovieID']
    
    # continuous_genre
    genres=[]
    for gen in items['Genres']:         
        genres.append(list(map(int, gen)))
        
    genres = pd.DataFrame(np.array(genres))
    genres['MovieID'] = items['MovieID']       
    
    # continuous_title
    titles = items.copy()
    titles = titles['Title']
    
    txt = BERT(list(titles),batch_size = 32)

    user_ctg, item_ctg, item_gen, item_txt = sample(ratings, u_categories,it_categories,genres, txt)
    train_loader, test_loader, valid_loader, y_scaler = data_split(user_ctg, item_ctg, item_gen, item_txt, ratings['Rating'], 
                                                                   config['data_loader']['args']['batch_size'])

    config['hyper_params']['user_count'] = user_count
    config['hyper_params']['item_count'] = item_count
    config['hyper_params']['genr_nfeat'] = len(item_gen[0])
    config['hyper_params']['txt_nfeat'] = len(item_txt[0])
    return train_loader, valid_loader, test_loader, y_scaler, config

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
        
        
