from data import BioASQPointwiseIterator, RankingIterator
from utils import EmptyEncodeBatch


class BioASQPointwiseIteratorForCasualLM(BioASQPointwiseIterator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_first_part = "Question: [QUESTION] \nDocument:"

        self.true_inputs_label = self.tokenizer("\n Relevant: true", add_special_tokens=False)
        self.false_inputs_label = self.tokenizer("\n Relevant: false", add_special_tokens=False)

        # TODO try replace \n by </s>
    
    def _tokenize(self, q_text, doc_text, label=None):
        """
        If doc is to big it needs to be truncated
        """
        q_inputs = self.tokenizer("Question: [QUESTION] \nDocument: ".replace("[QUESTION]", q_text), add_special_tokens=False) 
        d_inputs = self.tokenizer(doc_text, add_special_tokens=False)
        if label is not None:
            l_inputs = self.true_inputs_label if bool(label) else self.false_inputs_label
        else:
            l_inputs = EmptyEncodeBatch()

        required_space = 1 + len(q_inputs.input_ids) + len(l_inputs.input_ids) 
        
        d_inputs.input_ids = d_inputs.input_ids[:(self.max_length-required_space)]
        d_inputs.attention_mask = d_inputs.attention_mask[:(self.max_length-required_space)]
        
        input_ids = [2] + q_inputs.input_ids + d_inputs.input_ids + l_inputs.input_ids
        attention_mask = [1] + q_inputs.attention_mask + d_inputs.attention_mask + l_inputs.attention_mask
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
class RankingIteratorForCasualLM(RankingIterator, BioASQPointwiseIteratorForCasualLM):
    pass
    #@staticmethod
    #def get_n_samples(dataset):
        # BioASQPointwiseIteratorForCasualLM also implements this method, and since its static it will be the one
        # to get overrided
    #    return RankingIterator.get_n_samples(dataset)

