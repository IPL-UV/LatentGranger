import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import databases


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config, data_config, processing_mode):

        super().__init__()

        self.shuffle = config['shuffle']

        self.pin_memory = config['pin_memory']
        self.num_workers = config['num_workers']

        self.comfig = config 
        self.data_config = data_config

        data = getattr(databases, self.data_config['class']) 

        self.data_train = data(self.data_config, 'train', processing_mode) 
        self.data_val = data(self.data_config, 'val', processing_mode) 
        self.data_test = data(self.data_config, 'test', processing_mode)
        self.batch_size = self.data_config['batch_size']

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,
                              shuffle=self.shuffle, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers,
                              pin_memory=self.pin_memory)
