import argparse
import model.metric as module_metric
import model.loss as module_loss
from model.DeepRec import Rec_Model
from trainer.trainer import Trainer
from utils import Load_split_dataset
from utils.parse_config import ConfigParser
import torch

# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main(config):
    # load datasets
        
    # build model architecture, initialize weights, then print to console    
    model = Rec_Model(config['hyper_params'])
    #model.weights_init()  
    
    logger = config.get_logger('train') 
    logger.info(model)
    logger.info("-"*100)

    # get function handles of metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    criterion = getattr(module_loss, config['loss'])

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = Trainer(model, 
                      optimizer,
                      criterion,
                      metrics,
                      config,
                      train_set,
                      valid_set,
                      test_set)

    log = trainer.train()
    return log

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', type=str, 
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default= None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data', type=str)
    params = args.parse_args()
    
    config = ConfigParser.from_args(args)
    config, train_set, valid_set, test_set = Load_split_dataset(config)
    config['hyper_params']['user_count'] = user_count
    config['hyper_params']['item_count'] = item_count
    config['hyper_params']['genr_nfeat'] = genr_nfeat
    config['hyper_params']['txt_nfeat'] = txt_nfeat

    log = main(config)
