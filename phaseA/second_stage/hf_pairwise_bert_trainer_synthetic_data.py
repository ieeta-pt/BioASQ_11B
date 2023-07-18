import random
random.seed(42)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import json

from data import create_bioASQ_datasets, BioASQPointwiseIterator, RankingIterator, create_bioASQ_synthetic_dataset
from sampler import BasicSampler, HigherConfidenceNegativesSampler
from collator import RankingCollator

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from ranker_trainer import RankerTrainer
from metrics import RanxMetrics
from utils import setup_wandb, create_config

import torch

setup_wandb()
training_args = create_config("bert_trainer_config.yaml", 
                              output_dir="pubmed_bert_classifier_V2_syntheticV2",
                              dataloader_num_workers=1) # 10-15Gb

model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.model_max_length = 512

# 15_000
# 130_000

test_ds = create_bioASQ_datasets("../data/BioASQ-training11b/training11b_inflated_clean_wContents.jsonl", 
                                        "../first_stage/training_data_negatives.jsonl",
                                        tokenizer=tokenizer,
                                        subsets="test",
                                        train_iterator_class=BioASQPointwiseIterator[HigherConfidenceNegativesSampler],
                                        #train_max_questions=10,
                                        test_iterator_class=RankingIterator,
                                        test_split_percentage = 0.05,
                                        test_max_neg_docs=100, # 1000
                                        #test_max_questions=10,
                                        )

# use synthetic data for training

# uses 50 neg per question after the first 100 [100:1000] -> 50 random sample
train_ds = create_bioASQ_synthetic_dataset("../data/synthetic/v3/syn_question_positives.jsonl",
                                           "../data/synthetic/v3/syn_question_negatives.jsonl",
                                           "../data/synthetic/v3/syn_question_collection.jsonl",
                                           tokenizer=tokenizer,
                                           iterator_class=BioASQPointwiseIterator[BasicSampler],
                                           #max_questions=10,
                                           )

id2label = {0: "IRRELEVANT", 1: "RELEVANT"}
label2id = {"IRRELEVANT": 0, "RELEVANT": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)#.to("cuda")

trainer = RankerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    eval_data_collator=RankingCollator(tokenizer=tokenizer),
    preprocess_logits_for_metrics=lambda logits, labels: torch.nn.functional.softmax(logits, dim=-1)[:,1], #
    compute_metrics=RanxMetrics(test_ds.get_qrels()),
)

trainer.train()