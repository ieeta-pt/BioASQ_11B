import torch
import random

class ConcatIterator:
    
    def __init__(self, datasets, epoch, max_epoch):
        self.datasets = datasets
        self._iter_datasets = [iter(ds) for ds in datasets]
        self.weights = [len(ds) for ds in datasets]
        self.expected_number_of_samples = sum(self.weights )
        self.index = 0
        self.epoch = epoch
        self.max_epoch = max_epoch

    def __next__(self):
        
        if self.index>=self.expected_number_of_samples:
            raise StopIteration
        
        self.index += 1
        return next(random.choices(self._iter_datasets, weights=self.weights)[0])
    
class SynToTrueIterator(ConcatIterator):

    
    def __init__(self, datasets, epoch, max_epoch):
        super().__init__(datasets, epoch, max_epoch)

    #    self.expected_number_of_samples = len(self.datasets[0])
        true_prob = self.epoch/self.max_epoch
        syn_prob = 1-true_prob
        syn_datasets = len(self.datasets[1:])
        
        self.weights = [true_prob] + [syn_prob/syn_datasets for _ in range(syn_datasets)]
        print("CURRENT PROBS:", self.weights)
    
    

class BioASQDatasetConcat(torch.utils.data.IterableDataset):
    
    def __init__(self, datasets, iter_class=ConcatIterator, max_epoch=10):
        self.datasets = datasets
        self.iter_class = iter_class
        self.epoch = -1
        self.max_epoch = max_epoch
        
    def get_n_questions(self):
        return sum([ds.get_n_questions() for ds in self.datasets])
    
    def __len__(self):
        return sum([len(ds) for ds in self.datasets])
    
    def __iter__(self):
        #worker_info = torch.utils.data.get_worker_info()
        self.epoch += 1
        
        worker_info = torch.utils.data.get_worker_info()
        #print("WORKER_INFO!!!", worker_info)
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.iter_class(datasets=self.datasets, epoch=self.epoch, max_epoch=self.max_epoch)
        else:  # in a worker process
            # split workload
            assert worker_info.num_workers==1
           
            
            return self.iter_class(datasets=self.datasets, epoch=self.epoch, max_epoch=self.max_epoch)
        
