from pyserini.search.lucene import LuceneSearcher
import json
from collections import defaultdict
from tqdm import tqdm
from ranx import Qrels, Run
from ranx import evaluate
import multiprocessing as mp
import time
import click

def searcher_mp(out_queue, questions, baseline, identifier, k1,beta,fbnterms,fbdocs,qw):
    print(f"Searcher for Baseline {baseline} started")
    time.sleep(identifier)
    searcher = LuceneSearcher(f"../../data/indexes/baseline_{baseline}")
    
    searcher.set_bm25(k1, beta)
    
    if fbnterms:
        searcher.set_rm3(fbnterms, fbdocs, qw)
    
    
    for question in questions:
        hits = searcher.search(question["body"], k=1_000)
        #print(f"Searcher {identifier} took {es-ss} to search")
        out_queue.put({question["id"]: [{"text":json.loads(hit.raw)["contents"], "id":hit.docid, "score": hit.score} for hit in hits]})
        #print(f"Searcher {identifier} took {time.time()-es} to send")
    out_queue.put(None)


@click.command()
@click.option("--k1", default=1.2)
@click.option("--beta", default=0.5)
@click.option("--fbnterms", default=None)
@click.option("--fbdocs", default=None)
@click.option("--qw", default=None)
def main(k1,beta,fbnterms,fbdocs,qw):
    
    # load questions
    questions_by_baseline = defaultdict(list)
    total_num_questions = 0
    small_number = 0
    qrels_dict = {}
    with open("../../data/BioASQ-training11b/training11b_inflated_clean.jsonl")  as f:
        for line in f:
            data = json.loads(line)
            
            if int(data["baseline"])>=2016:
                questions_by_baseline[data["baseline"]].append(data)
                total_num_questions += 1
                qrels_dict[data["id"]] = {docid:1 for docid in data["documents"]}
                #small_number+=1
                #if small_number>10:
                #    break

    main_thread_queue = mp.JoinableQueue(100)
    
    processes = []
    timing = 0
    for i,(b,questions) in enumerate(questions_by_baseline.items()):
        processes.append(mp.Process(target=searcher_mp, args=(main_thread_queue, questions[:len(questions)//2], b, timing, k1,beta,fbnterms,fbdocs,qw)) )
        timing +=1 
        processes.append(mp.Process(target=searcher_mp, args=(main_thread_queue, questions[len(questions)//2:], b, timing, k1,beta,fbnterms,fbdocs,qw)) )
        timing +=1 
    
    for process in processes:
        process.start()
    
    
    #print("Collect in the main thread")
    c=0
    with open("../../data/BioASQ-training11b/training_data_negatives.jsonl","w") as fOut:
        with tqdm(total=total_num_questions) as pbar:
            while True:
                #print("get from queue")
                data = main_thread_queue.get()
                main_thread_queue.task_done()
                if data == None:
                    c+=1
                    if c>=len(processes):
                        break
                    continue
                
                # remove positive samples
                for q_id in data.keys():
                    neg_docs = [doc for doc in data[q_id] if doc["id"] not in qrels_dict[q_id]]
     
                    out = {"id":q_id, "neg_docs":neg_docs}
                    fOut.write(f"{json.dumps(out)}\n")

                pbar.update(1)


if __name__ == '__main__':
    mp.set_start_method("forkserver") #fork spawn
    main()
    #for i in range(len(hits)):
#    print(f"{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}")
#And if you want to get more info about the content .. you should use json.loads(hits[i].raw)["content"]
