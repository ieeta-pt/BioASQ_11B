#!/bin/bash
echo "Recreating submission batch 02 of Bit.UA in BioASQ 11B phase A"

echo "Preparing Run 00 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-30580 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/ 

python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-27531 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-24472 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/ 


echo "Preparing Run 01 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V1-checkpoint-32230 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V1-checkpoint-24472 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-30580 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-27531 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-24472 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/ 



echo "Preparing Run 02 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-29268-finetune-checkpoint-9174 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-10722 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-12236 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-12236 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-5397 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/



echo "Preparing Run 03 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/

echo "Preparing Run 04 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-30580 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-27531 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-24472 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V1_checkpoint-32230 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V1_checkpoint-25784 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-30580 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-27531 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6_checkpoint-24472 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-29268-finetune-checkpoint-9174 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-10722 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-12236 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-12236 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V2_synthetic-checkpoint-32490-finetune-checkpoint-5397 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 500 --path_to_save Batch02/runs/
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch02/runs/
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290 --baseline_path ../first_stage/Batch02/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch02/runs/


echo "Converting to relevance"
python relevance_converter.py --run_paths="Batch02/runs/*"



echo "Fusing the results"
python fusion.py Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-30580_1000.json Batch02/runs/T-Almeida/BioASQ-11B_V6-checkpoint-27531_1000.json Batch02/runs/T-Almeida/BioASQ-11B_V6-checkpoint-24472_1000.json --out Batch02/runs/ranx_run0.json --method rrf
python fusion.py Batch02/runs/T-Almeida/BioASQ-11B_V1-checkpoint-32230_100.json Batch02/runs/T-Almeida/BioASQ-11B_V1-checkpoint-24472_100.jsonBatch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-30580_100.jsonBatch02/runs/T-Almeida/BioASQ-11B_V6-checkpoint-27531_100.jsonBatch02/runs/T-Almeida/BioASQ-11B_V6-checkpoint-24472_100.json --out Batch02/runs/ranx_run1.json --method rrf
python fusion.py Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-29268-finetune-checkpoint-9174_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-10722_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-11475_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-11475_500.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-12236_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-12236_500.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-5397_100.json --out Batch02/runs/ranx_run2.json --method rrf
python fusion.py Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794_500.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023_500.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475_500.json --out Batch02/runs/ranx_run3.json --method rrf
python fusion.py Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-30580_1000.json Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-27531_1000.json Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-24472_1000.json Batch02/runs/T-Almeida/BioASQ-11B_V1_checkpoint-32230_100.json Batch02/runs/T-Almeida/BioASQ-11B_V1_checkpoint-25784_100.json Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-30580_100.json Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-27531_100.json Batch02/runs/T-Almeida/BioASQ-11B_V6_checkpoint-24472_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-29268-finetune-checkpoint-9174_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-10722_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-11475_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-11475_500.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-12236_100.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-12236_500.json Batch02/runs/T-Almeida/BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-5397_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10023_500.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-10794_500.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475_100.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475_500.json Batch02/runs/T-Almeida/BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710_100.json Batch02/runs/T-Almeida/BioASQ-11B_large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244_100.json Batch02/runs/T-Almeida/BioASQ-11B_large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244_1000.json Batch02/runs/T-Almeida/BioASQ-11B_large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290_100.json --out Batch02/runs/ranx_run4.json --method rrf


echo "Convert to bioasq format"
python bioasq_format_converter.py Batch02/runs/ranx_run0.json
python bioasq_format_converter.py Batch02/runs/ranx_run1.json
python bioasq_format_converter.py Batch02/runs/ranx_run2.json
python bioasq_format_converter.py Batch02/runs/ranx_run3.json
python bioasq_format_converter.py Batch02/runs/ranx_run4.json