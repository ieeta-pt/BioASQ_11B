from data import BioASQPointwiseIterator, RankingIterator, InferenceRankingIterator
from utils import EmptyEncodeBatch


class BioASQPointwiseIteratorForConditionalGeneration(BioASQPointwiseIterator): #tokenizer #next
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        self.prompt_first_part = "Query: [QUESTION] Document: "
        self.prompt_last_part = " Relevance:"

        with self.tokenizer.as_target_tokenizer():
            self.true_inputs_label = self.tokenizer("true")
            self.false_inputs_label = self.tokenizer("false")

        # TODO try replace \n by </s>
    
    def _tokenize(self, q_text, doc_text, label=None):
        """
        If doc is to big it needs to be truncated
        """
        q_inputs = self.tokenizer(self.prompt_first_part.replace("[QUESTION]", q_text), add_special_tokens=False) 
        d_inputs = self.tokenizer(doc_text, add_special_tokens=False)
        last_part_inputs = self.tokenizer(self.prompt_last_part, add_special_tokens=False)
        

        required_space = 1 + len(q_inputs.input_ids) + len(last_part_inputs.input_ids)
        
        d_inputs.input_ids = d_inputs.input_ids[:(self.max_length-required_space)]
        d_inputs.attention_mask = d_inputs.attention_mask[:(self.max_length-required_space)]
        
        input_ids = q_inputs.input_ids + d_inputs.input_ids + last_part_inputs.input_ids + [1]
        attention_mask = q_inputs.attention_mask + d_inputs.attention_mask + last_part_inputs.attention_mask + [1]
        
        if label is not None:
            l_inputs = self.true_inputs_label if bool(label) else self.false_inputs_label
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": l_inputs.input_ids}
        else:
            return {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": [0]}
    
class RankingIteratorForConditionalGeneration(RankingIterator, BioASQPointwiseIteratorForConditionalGeneration):
    pass
    #@staticmethod
    #def get_n_samples(dataset):
        # BioASQPointwiseIteratorForCasualLM also implements this method, and since its static it will be the one
        # to get overrided
    #    return RankingIterator.get_n_samples(dataset)
    
class InferenceRankingIteratorForConditionalGeneration(InferenceRankingIterator, BioASQPointwiseIteratorForConditionalGeneration):
    pass

