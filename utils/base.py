import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader
import torch

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
