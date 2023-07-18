from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from ranker_trainer import RankerTrainer, RankingEvalPrediction
from metrics import RanxMetrics
from data import create_bioASQ_datasets, RankingIterator, BioASQDataset, InferenceRankingIterator
from collator import RankingCollator
from utils import create_config, load_rank_data

from collections import defaultdict
import torch
import os
from ranx import Qrels, Run
import click
import re

@click.command()
@click.option("--checkpoint")
@click.option("--revision", default="")
@click.option("--baseline_path")
@click.option("--at", default=100)
@click.option("--qrels_path", default=None)
@click.option("--path_to_save", default=None)
@click.option('--pairwise', is_flag=True)
def main(checkpoint, revision, baseline_path, at, qrels_path, path_to_save, pairwise):
    print("Is pairwise?", pairwise)
    training_args = create_config("bert_trainer_config.yaml", 
                                output_dir="../../HF_CACHE/.dummy",
                                per_device_eval_batch_size=128,
                                dataloader_num_workers=2,
                                report_to="none",
                                )
    
    model_checkpoint = checkpoint

    if revision!="":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                revision=revision,
                                                                cache_dir="../../HF_CACHE")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                              revision=revision,
                                              cache_dir="../../HF_CACHE")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.model_max_length = 512

    if qrels_path is None:
        qrels = None
        split_name = None
    else:
        split_name = str(re.findall(r"s([0-9].[0-9]+)", qrels_path)[0])
        qrels = Qrels.from_file(qrels_path).to_dict()
        

    # load bm25 rerank data
    dataset = load_rank_data(baseline_path, at=at, qrels=qrels)
    
    test_ds = BioASQDataset(dataset,
                            tokenizer, 
                            iterator_class = InferenceRankingIterator, 
                            #max_questions=10,
                            qrels_dict = qrels)

    print("Number of questions", test_ds.get_n_questions())
        
        

    #print("Qrels", test_ds.get_qrels())

    # lambda logits, labels: torch.nn.softmax(logits,dim=-1)[:,1]


    def bert_preprocess_pairwise(logits, labels):
        return logits[:,0]

    def bert_preprocess_pointwise(logits, labels):
        return torch.nn.functional.softmax(logits,dim=-1)[:,1]
    
    if pairwise:
        logits_f = bert_preprocess_pairwise
    else:
        logits_f = bert_preprocess_pointwise
    
    trainer = RankerTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=logits_f,
        compute_metrics=None,
        data_collator=RankingCollator(tokenizer=tokenizer),
    )
    
    #print("Dataset", next(iter(test_ds)).keys())
    
    #test_dataloader = trainer.get_test_dataloader(test_ds)
    #print("Test DL",next(iter(test_dataloader)).keys())
    
    #test_dataloader = trainer.get_eval_dataloader(test_ds)
    #print("Eval DL",next(iter(test_dataloader)).keys())
    
    #exit()

    evaluationOutput = trainer.predict(test_ds)
    
    run_dict = defaultdict(dict)
    for i, metadata in enumerate(evaluationOutput.ranking_metadata):
        run_dict[metadata["id"]][metadata["doc_id"]] = evaluationOutput.predictions[i]
    
    #qrels = Qrels({k:v for k,v in self.qrels_dict.items() if k in run_dict})

    run = Run(run_dict)
    flat_name = checkpoint.replace("/","_")
    s_name = "" if split_name is None else split_name
    
    name_begining = f"ranx_{flat_name}"
    
    if revision!="":
        name_begining = f"{name_begining}_{revision}"
    
    if split_name:
        run.save(os.path.join(path_to_save,f"{name_begining}_{at}_test_split{s_name}.json"))
    else:
        run.save(os.path.join(path_to_save,f"{name_begining}_{at}.json"))

    #print(trainer.evaluate(test_ds))
    
if __name__ == '__main__':
    main()