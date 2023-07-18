from pyserini.search.lucene import LuceneSearcher
import json
from collections import defaultdict
from tqdm import tqdm
from ranx import Qrels, Run
from ranx import evaluate
import multiprocessing as mp
import time
import click
import os

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
        out_queue.put({"id":question["id"],
                       "documents": [{"id":hit.docid, "score":hit.score, "text":json.loads(hit.raw)["contents"]} for hit in hits],
                       "question": question["body"],
                       })
        #print(f"Searcher {identifier} took {time.time()-es} to send")
    out_queue.put(None)

def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

@click.command()
@click.option("--testset")
@click.option("--proc", default=10)
@click.option("--k1", default=1.2)
@click.option("--beta", default=0.5)
@click.option("--fbnterms", default=None)
@click.option("--fbdocs", default=None)
@click.option("--qw", default=None)
def main(testset, proc, k1,beta,fbnterms,fbdocs,qw):
    
    batch_folder = os.path.dirname(testset)
    
    # load questions
    questions = []
    total_num_questions = 0
    small_number = 0
    qrels_dict = {}
    with open(testset)  as f:
        test_dataset = json.load(f)["questions"]
        for data in test_dataset:

            questions.append(data)
            total_num_questions += 1
                #small_number+=1
                #if small_number>10:
                #    break

    questions_chunks = split_chunks(questions, proc)
    
    run= []
    main_thread_queue = mp.JoinableQueue(100)
    
    processes = []
    timing = 0
    for i, chunk_question in enumerate(questions_chunks):
        processes.append(mp.Process(target=searcher_mp, args=(main_thread_queue, chunk_question, 2023, i, k1,beta,fbnterms,fbdocs,qw)) )
    
    for process in processes:
        process.start()
    
    
    #print("Collect in the main thread")
    c=0
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
            run.append(data)
            pbar.update(1)
        #print("update")
    
    #print("Evaluation Start")
    with open(os.path.join(batch_folder, f"bm25_{k1}_{beta}.jsonl"), "w") as fOut:
        for q_data in run:
            fOut.write(f"{json.dumps(q_data)}\n")
    
if __name__ == '__main__':
    mp.set_start_method("forkserver") #fork spawn
    main()
    #for i in range(len(hits)):
#    print(f"{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}")
#And if you want to get more info about the content .. you should use json.loads(hits[i].raw)["content"]
