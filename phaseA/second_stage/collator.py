
class RankingCollator:
    
    def __init__(self, 
                 tokenizer, 
                 model_inputs={"input_ids", "attention_mask", "token_type_ids"}, 
                 padding=True,
                 max_length=None):
        self.tokenizer = tokenizer
        self.model_inputs = model_inputs
        self.padding = padding
        self.max_length = max_length
        
    def __call__(self, batch):
        batch = {key: [i[key] for i in batch] for key in batch[0]}

        reminder_keys = set(batch.keys())-self.model_inputs
        return {"inputs": self.tokenizer.pad({k:batch[k] for k in self.model_inputs},
                                     padding=self.padding,
                                     max_length=self.max_length,
                                     return_tensors="pt")
                } | {k:batch[k] for k in reminder_keys}

class PairwiseCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        batch = {key: [i[key] for i in batch] for key in batch[0]}
        #print(batch.keys())
        return {
            "pos_doc": self.tokenizer.pad(batch["pos_doc"],
                                     padding=True,
                                     return_tensors="pt"),
            "neg_doc": self.tokenizer.pad(batch["neg_doc"],
                                     padding=True,
                                     #max_length=512,
                                     return_tensors="pt")
            }


class RankingCollatorForCasualLM(RankingCollator):
    def __init__(self, tokenizer, model_inputs={"input_ids", "attention_mask"}):
        super().__init__(tokenizer, model_inputs=model_inputs)
        
class RankingCollatorForSeq2Seq(RankingCollator):
    def __init__(self, tokenizer, model_inputs={"input_ids", "attention_mask", "decoder_input_ids"}):
        super().__init__(tokenizer, model_inputs=model_inputs)