import numpy as np
import torch
from trainer.base_trainer import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import compute_variances

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, 
                      optimizer,
                      criterion,
                      metric_ftns,
                      config,
                      train_set,
                      valid_set,
                      test_set,
                      y_scaler):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.batch_size = config['data_loader']['batch_size']
        self.criterion = criterion
        self.y_scaler = y_scaler
                          
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        
        self.train_n_batches = int(np.ceil(float(config["data_loader"]["n_train"]) / float(self.batch_size)))
        self.valid_n_batches = int(np.ceil(float(config["data_loader"]["n_valid"]) / float(self.batch_size)))
        self.test_n_batches = int(np.ceil(float(config["data_loader"]["n_test"]) / float(self.batch_size)))

        self.do_validation = self.valid_set is not None
        self.lr_scheduler = optimizer
        self.log_step = 4  # reduce this if you want more logs

        self.metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):  
        self.model.train()
        self.metrics.reset()                
        outs = np.array([]), trgs = np.array([])
        for batch_idx, (user,cat,genr,txt,y) in enumerate(self.train_set):
            user, cat, genr, txt = user.to(self.device), cat.to(self.device), genr.to(self.device), txt.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(user,cat,genr,txt)
            loss = self.criterion(output, y, self.device)
            loss.backward()
            self.optimizer.step()
            self.metrics.update('loss', loss.item())
            
            if idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(idx),
                    loss.item(),
                ))
            output = self.y_scaler.inverse_transform(output.cpu().detach().numpy())
            y = self.y_scaler.inverse_transform(y.cpu().detach().numpy())
            outs = np.append(outs, output)
            trgs = np.append(trgs, y)
            
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
                
        log = self.metrics.result()         
        if self.do_validation:
            val_log = self._infer(self.valid_set)
            log.update(**{'val_' + k: v for k, v in val_log.items()})      
        return log

    def _infer(self, data_set):
        self.model.eval()
        self.metrics.reset()
        with torch.no_grad():
            outs = np.array([]), trgs = np.array([])
            for batch_idx, (user,cat,genr,txt,y) in enumerate(data_set):
                user, cat, genr, txt = user.to(self.device), cat.to(self.device), genr.to(self.device), txt.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(user,cat,genr,txt)
                loss = self.criterion(output, y, self.device)
                loss.backward()
                self.optimizer.step()
                self.metrics.update('loss', loss.item())  
                
                output = self.y_scaler.inverse_transform(output.cpu().detach().numpy())
                y = self.y_scaler.inverse_transform(y.cpu().detach().numpy())
                outs = np.append(outs, output)
                trgs = np.append(trgs, y)
                
                
        for met in self.metric_ftns:
            self.metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
                
        log = self.metrics.result()
            
        return log
        

    def _test_epoch(self):
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])
        self.model.eval()
        
        log = {}        
        train_log = self._infer(self.train_set)
        valid_log = self._infer(self.valid_set)
        test_log = self._infer(self.test_set)
        log.update(**{'train_' + k: v for k, v in train_log.items()})
        log.update(**{'val_' + k: v for k, v in valid_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
                        
        self.logger.info('='*100)
        self.logger.info('Inference is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:20s}: {}'.format(str(key), value))  
        self.logger.info('='*100)

        return log
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx * self.batch_size
        total = self.config['data_loader']['n_train']
        return base.format(current, total, 100.0 * current / total)
        
        
