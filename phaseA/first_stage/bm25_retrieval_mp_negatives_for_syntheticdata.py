from pyserini.search.lucene import LuceneSearcher
import json
from collections import defaultdict
from tqdm import tqdm
from ranx import Qrels, Run
from ranx import evaluate
import multiprocessing as mp
import time
import click

def searcher_mp(out_queue, questions, identifier, k1,beta,fbnterms,fbdocs,qw):
    print(f"Searcher started")
    time.sleep(identifier)
    searcher = LuceneSearcher(f"../../data/indexes/baseline_2022")
    
    searcher.set_bm25(k1, beta)
    
    if fbnterms:
        searcher.set_rm3(fbnterms, fbdocs, qw)
    
    
    for question in questions:
        hits = searcher.search(question["body"], k=1_000)
        #print(f"Searcher {identifier} took {es-ss} to search")
        out_queue.put({question["id"]: [{"text":json.loads(hit.raw)["contents"], "id":hit.docid, "score": hit.score} for hit in hits]})
        #print(f"Searcher {identifier} took {time.time()-es} to send")
    out_queue.put(None)

def split_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

@click.command()
@click.option("--proc", default=40)
@click.option("--k1", default=1.2)
@click.option("--beta", default=0.5)
@click.option("--fbnterms", default=None)
@click.option("--fbdocs", default=None)
@click.option("--qw", default=None)
def main(proc,k1,beta,fbnterms,fbdocs,qw):
    
    # load questions
    training_questions = []
    total_num_questions = 0
    small_number = 0
    qrels_dict = {}
    with open("../../data/BioASQ-training11b/synthetic_questions_top_100.jsonl")  as f:
        for line in f:
            data = json.loads(line)
            training_questions.append(data)
            total_num_questions += 1
            qrels_dict[data["id"]] = {doc["id"]:1 for doc in data["documents"]} # gs
                #small_number+=1
                #if small_number>10:
                #    break

    training_questions_chunks = split_chunks(training_questions, proc)
    
    main_thread_queue = mp.JoinableQueue(100)
    
    processes = []
    timing = 0
    # GIL -> global interpreter lock
    for i, chunk_question in enumerate(training_questions_chunks):
        processes.append(mp.Process(target=searcher_mp, args=(main_thread_queue, chunk_question, i, k1,beta,fbnterms,fbdocs,qw)) )
    
    for process in processes:
        process.start()
    
    
    #print("Collect in the main thread")
    c=0
    with open("../../data/BioASQ-training11b/training_data_negatives_synthetic.jsonl","w") as fOut:
        with tqdm(total=total_num_questions) as pbar:
            while True:
                #print("get from queue")
                data = main_thread_queue.get()
                main_thread_queue.task_done()
                
                # terminal stoping the while loop
                if data == None:
                    c+=1
                    if c>=len(processes):
                        break
                    continue
                
                # remove positive samples
                for q_id in data.keys():
                    neg_docs = [doc for doc in data[q_id] if doc["id"] not in qrels_dict[q_id]]
                    
                    # run monoT5?
                    # get the ones with low prob < 0.000
                    
                    out = {"id":q_id, "neg_docs":neg_docs[125:]}
                    fOut.write(f"{json.dumps(out)}\n")

                pbar.update(1)


if __name__ == '__main__':
    mp.set_start_method("forkserver") #fork spawn
    main()
    #for i in range(len(hits)):
#    print(f"{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}")
#And if you want to get more info about the content .. you should use json.loads(hits[i].raw)["content"]
