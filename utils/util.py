import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
from utils.base import load_data, transform_category
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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

def sample(data,user,item,genre,txt):
    user_,item_,genre_,txt_ = [],[],[],[]    
    for i in range(len(data)):
        UserID = data.iloc[i,:]['UserID']
        MovieID = data.iloc[i,:]['MovieID']
        user_.append(list(user[user['UserID']== UserID].drop(['UserID'], axis='columns').iloc[0,:]))
        item_.append(list(item[item['MovieID'] == MovieID].drop(['MovieID'], axis='columns').iloc[0,:]))
        genre_.append(list(genre[genre['MovieID'] == MovieID].drop(['MovieID'], axis='columns').iloc[0,:]))
        txt_.append(list(txt[txt['MovieID'] == MovieID].drop(['MovieID'], axis='columns').iloc[0,:]))
            
    user_   = np.array(user_).reshape(len(data),-1)
    item_   = np.array(item_).reshape(len(data),-1)
    genre_  = np.array(genre_).reshape(len(data),-1)
    txt_    = np.array(txt_).reshape(len(data),-1)            
    
    return user_, item_, genre_, txt_



def data_split(user_ctg, item_ctg, item_gen, item_txt, label, batch_size):    
    indices = np.random.permutation(np.arange(len(ratings)))
    train_idx = indices[:81000]
    test_idx = indices[81000:91000]
    valid_idx = indices[91000:]
    
    label    = np.array(label)
    item_txt = np.array(item_txt)
    
    y_scaler = MinMaxScaler()
    y_scaler.fit(np.array(label[train_idx].reshape(-1,1)))
    label   = y_scaler.transform(label.reshape(-1,1))  
    
    txt_scaler = StandardScaler()
    txt_scaler.fit(np.array(item_txt[train_idx]))
    item_txt = standard.transform(item_txt)

    user_ctg    = torch.LongTensor(np.array(user_ctg))
    item_ctg    = torch.LongTensor(np.array(item_ctg))
    item_gen    = torch.FloatTensor(np.array(item_gen))
    item_txt    = torch.FloatTensor(np.array(item_txt))
    label       = torch.FloatTensor(np.array(label))

    train = TensorDataset(user_ctg[train_idx], item_ctg[train_idx], 
                          item_gen[train_idx],   item_txt[train_idx], label[train_idx])
    test  = TensorDataset(user_ctg[test_idx], item_ctg[test_idx],
                          item_gen[test_idx],   item_txt[test_idx], label[test_idx])        
    valid  = TensorDataset(user_ctg[valid_idx], item_ctg[valid_idx],
                          item_gen[valid_idx],   item_txt[valid_idx], label[valid_idx])       

    train_load  = torch.utils.data.DataLoader(dataset= train,batch_size=batch_size,shuffle=False,drop_last=True) 
    test_load   = torch.utils.data.DataLoader(dataset= test,batch_size=batch_size,shuffle=False,drop_last=True) 
    valid_load   = torch.utils.data.DataLoader(dataset= valid,batch_size=batch_size,shuffle=False,drop_last=True) 
    
    return train_load, test_load, valid_load, y_scaler

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
        
        
