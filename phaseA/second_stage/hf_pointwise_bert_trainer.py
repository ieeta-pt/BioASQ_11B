from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import json

from data import create_bioASQ_datasets, BioASQPointwiseIterator, RankingIterator
from sampler import BasicSampler, HigherConfidenceNegativesSampler
from collator import RankingCollator

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from ranker_trainer import RankerTrainer
from metrics import RanxMetrics
from utils import setup_wandb, create_config

import torch
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("checkpoint", type=str)
args = parser.parse_args()

model_checkpoint = args.checkpoint#"pubmed_bert_classifier_V2_synthetic/checkpoint-29268"

setup_wandb()
training_args = create_config("bert_trainer_config.yaml", 
                              output_dir=model_checkpoint.replace("/", "-")+"-finetune",
                              dataloader_num_workers=4,
                              num_train_epochs=5,
                              per_device_train_batch_size= 8)

#model_checkpoint = "pubmed_bert_classifier_V2_synthetic/checkpoint-32490"
#model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.model_max_length = 512

train_ds, test_ds = create_bioASQ_datasets("../../data/BioASQ-training11b/training11b_inflated_clean_wContents.jsonl", 
                                        "../../data/BioASQ-training11b/training_data_negatives.jsonl",
                                        tokenizer=tokenizer,
                                        subsets=["train", "test"],
                                        train_iterator_class=BioASQPointwiseIterator[HigherConfidenceNegativesSampler],
                                        #train_max_questions=300,
                                        test_iterator_class=RankingIterator,
                                        test_split_percentage = 0.05,
                                        test_max_neg_docs=100,
                                        #test_max_questions=10,
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