from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import json

from data import create_bioASQ_datasets, BioASQPairwiseIterator, RankingIterator
from sampler import BasicSampler
from collator import RankingCollator, PairwiseCollator
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from ranker_trainer import RankerTrainer, PairwiseTrainer
from metrics import RanxMetrics
from utils import setup_wandb, create_config
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("checkpoint", type=str)
args = parser.parse_args()

model_checkpoint = args.checkpoint


setup_wandb()
training_args = create_config("bert_trainer_config.yaml", 
                              output_dir=model_checkpoint.replace("/", "-")+"-finetune",
                              per_device_train_batch_size=4)

#model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_ds, test_ds = create_bioASQ_datasets("../data/BioASQ-training11b/training11b_inflated_clean_wContents.jsonl", 
                                        "../first_stage/training_data_negatives.jsonl",
                                        tokenizer=tokenizer,
                                        subsets=["train", "test"],
                                        train_iterator_class=BioASQPairwiseIterator[BasicSampler],
                                        #train_max_questions=10,
                                        test_iterator_class=RankingIterator,
                                        test_split_percentage = 0.05,
                                        test_max_neg_docs=100,
                                        #test_max_questions=10,
                                        )

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=1
)

trainer = PairwiseTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=PairwiseCollator(tokenizer=tokenizer),
    eval_data_collator=RankingCollator(tokenizer=tokenizer),
    preprocess_logits_for_metrics=lambda logits, labels: logits[:,0],
    compute_metrics=RanxMetrics(test_ds.get_qrels()),
)

trainer.train()