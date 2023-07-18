import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

import click
import json
from tqdm import tqdm

checkpoint_model = "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

model = AutoModelForCausalLM.from_pretrained(checkpoint_model,
                                             torch_dtype=torch.float16,
                                             load_in_8bit=True, 
                                             device_map="auto",
                                             cache_dir="../HF_CACHE"
                                             )

tokenizer = AutoTokenizer.from_pretrained(checkpoint_model, cache_dir="../HF_CACHE")

def generate_prompt_OA(text, question):
    
    return f"""<|prompter|>Context: \"{text}\"

Question: \"{question}\"

Short and concise answer: <|endoftext|><|assistant|>"""


def evaluate(
    prompt,
):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    generation_config = GenerationConfig(top_p = 0.95,
                                     temperature=0.85,
                                     top_k = 50,
                                     repetition_penalty=1.2,
                                     num_beams = 1,
                                     #length_penalty=-1,
                                     do_sample=True,
                                     max_new_tokens=256)

    seq_len = len(inputs.input_ids[0])
    outs = model.generate(**inputs, 
                    generation_config = generation_config,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)
    #print(outs[0].shape) #[Number]
    return tokenizer.decode(outs[0][seq_len:].cpu()).strip()

@click.command()
@click.argument("testset")
@click.argument("doc_best_file")
@click.option("model_type", default="oa-pythia")
def main(testset, doc_best_file, model_type):

    
    if model_type.startswith("oa"):
        generate_prompt = generate_prompt_OA
    
    questions = {}
    with open(testset) as f:
        for q_data in json.load(f)["questions"]:
            questions[q_data["id"]] = q_data

    doc_list = []
    with open(f"{doc_best_file}") as f:
        for line in f:
            doc = json.loads(line)
            doc["pmid"] = doc["id"]
            del doc["id"]
            doc_list.append(doc|questions[doc["query_id"]])

    out_file, _ = os.path.splitext(doc_best_file)
    
    with open(f"{out_file}_wAnswer.jsonl", "w") as fOut:

        for doc in tqdm(doc_list):
            prompt = generate_prompt(model_type, doc["text"], doc["body"])
            doc["awnser"] = evaluate(prompt)
            fOut.write(f"{json.dumps(doc)}\n")
            
if __name__ == "__main__":
    main()