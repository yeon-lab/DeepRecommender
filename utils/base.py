import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def load_data(file_name, split):
    data = open(file_name, encoding = "ISO-8859-1" )
    data = data.read().split('\n')
    data = [i.split(split) for i in data]
    return data

def transform_category(data):
    Dicts = [{} for _ in range(data.shape[1])]
    for user in np.array(data):
        for j in range(data.shape[1]):
            Dicts[j][user[j]] = 1
    for i in range(data.shape[1]):
        for j, key in enumerate(list(Dicts[i].keys())):
            Dicts[i][key] = j        
    for j in range(data.shape[1]):
        for i, x in enumerate(data.iloc[:,j]):
            data.iloc[i,j] = Dicts[j][x] 
    nfeat = [len(i) for i in Dicts]
    return data, nfeat

def padding(inlist):
    padding_list=[]
    for i in inlist:
        padding_list.append([0]*(512-len(i))+i)
    return padding_list

def data_preprocessing(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(torch.device("cuda:0"))
    
    tokenized_text=[tokenizer.tokenize(i) for i in text]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    segments_ids=[[0]*len(i) for i in tokenized_text]
    
    tokens_tensor = torch.tensor(padding(indexed_tokens)).to(torch.device("cuda:0"))
    segments_tensors = torch.tensor(padding(segments_ids)).to(torch.device("cuda:0"))
    return tokens_tensor, segments_tensors

def BERT(titles, batch_size=20):
    tokens_tensor,segments_tensors = data_preprocessing(titles)
    tokens_loader = DataLoader(dataset=tokens_tensor,batch_size=batch_size,shuffle=False,drop_last=False)
    segments_loader = DataLoader(dataset=segments_tensors,batch_size=batch_size,shuffle=False,drop_last=False)
    print('preprocessing text------------')
    with torch.no_grad():
        i=0
        for tokens, segments in zip(tokens_loader,segments_loader):
            a,b=model(tokens, segments)
            if i==0:
                output_=b.cpu().detach().numpy()
            else:
                output_=np.append(output_, b.cpu().detach().numpy(), axis=0)
            print((i+1)*batch_size,'-th data!')
            i+=1
            del tokens, segments 
    print('preprocessing text completed-------')
    return output_



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
