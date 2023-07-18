import random

class BasicSampler:
    def __init__(self, slice_dataset, collection, *args, **kwargs):
        self.slice_dataset = slice_dataset
        self.collection = collection
        self.q_ids = list(self.slice_dataset.keys())
    
    def choose_question(self, sample_index, epoch):
        q_id = random.choice(self.q_ids)
        q_text = self.slice_dataset[q_id]["question"]
        return q_id, q_text
    
    def choose_positive_doc(self, sample_index, epoch, q_id):
        if self.collection:
            pos_pmid = random.choice(self.slice_dataset[q_id]["pos_docs"])
            return self.collection[pos_pmid]
        else:
            return random.choice(self.slice_dataset[q_id]["pos_docs"])["text"] 
    
    def choose_negative_doc(self, sample_index, epoch, q_id):
        if self.collection:
            neg_pmid = random.choice(self.slice_dataset[q_id]["neg_docs"])["id"]
            return self.collection[neg_pmid]
        else:
            return random.choice(self.slice_dataset[q_id]["neg_docs"])["text"]
            

class HigherConfidenceNegativesSampler(BasicSampler):
    
    def choose_negative_doc(self, sample_index, epoch, q_id):
        if self.collection:
            neg_pmid = random.choice(self.slice_dataset[q_id]["neg_docs"][10:])["id"]
            return self.collection[neg_pmid]["text"]
        else:
            return random.choice(self.slice_dataset[q_id]["neg_docs"][10:])["text"] 


class ShifterSampler:
    def __init__(self, slice_dataset, max_epoch, *args, **kwargs):
        self.slice_dataset = slice_dataset
        self.max_epoch = max_epoch
        self.q_ids = list(self.slice_dataset.keys())
    
    def choose_question(self, sample_index, epoch):
        q_id = random.choice(self.q_ids)
        q_text = self.slice_dataset[q_id]["question"]
        return q_id, q_text
    
    def choose_positive_doc(self, sample_index, epoch, q_id):
        return random.choice(self.slice_dataset[q_id]["pos_docs"])["text"]
    
    def choose_negative_doc(self, sample_index, epoch, q_id):
        neg_docs = self.slice_dataset[q_id]["neg_docs"]
        interval = len(neg_docs)//(self.max_epoch+1)
        return random.choice(neg_docs[interval*epoch:])["text"]