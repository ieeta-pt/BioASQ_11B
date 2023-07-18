#!/bin/bash
echo "Recreating submission batch 01 of Bit.UA in BioASQ 11B phase A"

echo "Preparing Run 00 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-30580 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-27531 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-24472 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch01/runs/ 


echo "Preparing Run 01 rerank"
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V1-checkpoint-32230 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V1-checkpoint-25784 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-30580 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-27531 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-24472 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 


echo "Preparing Run 02 rerank"
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision base_t5_base_monot5_classifier_V1-checkpoint-29065 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision base_t5_base_monot5_classifier_V1-checkpoint-11626 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 


echo "Preparing Run 03 rerank"
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-30580 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_bert.py --checkpoint T-Almeida/BioASQ-11B --revision V6-checkpoint-27531 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 


echo "Preparing Run 04 rerank"
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 1000 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 
python hf_rerank_t5.py --checkpoint T-Almeida/BioASQ-11B --revision large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290 --baseline_path ../first_stage/Batch01/bm25_0.5_0.3.jsonl --at 100 --path_to_save Batch01/runs/ 



echo "Converting to relevance"
python relevance_converter.py --run_paths="Batch01/runs/*"



echo "Fusing the results"
python fusion.py Batch01/runs/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-30580_1000.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V6-V6-checkpoint-27531_1000.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-24472_1000.json --out Batch01/runs/ranx_run0.json --method rrf
python fusion.py Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-32230_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-V1-checkpoint-25784_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-30580_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V6-checkpoint--27531_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-24472_100.json --out Batch01/runs/ranx_run1.json --method rrf
python fusion.py Batch01/runs/ranx_T-Almeida_BioASQ-11B_base_t5_base_monot5_classifier_V1-checkpoint-29065_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_base_t5_base_monot5_classifier_V1-checkpoint-11626_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244_100.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290_100.json --out Batch01/runs/ranx_run2.json --method rrf
python fusion.py Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244_100_relevance.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290_100_relevance.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-30580_100_relevance.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-30580_100_relevance.json --out Batch01/runs/ranx_run3.json --method rrf
python fusion.py Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244_1000_relevance.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290_1000_relevance.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-12244_100_relevance.json Batch01/runs/ranx_T-Almeida_BioASQ-11B_V1-large_nobf__t5_large_monot5_classifier_V1_noBF_test-0.05-checkpoint-15290_100_relevance.json --out Batch01/runs/ranx_run4.json --method rrf

echo "Convert to bioasq format"
python bioasq_format_converter.py Batch01/runs/ranx_run0.json
python bioasq_format_converter.py Batch01/runs/ranx_run1.json
python bioasq_format_converter.py Batch01/runs/ranx_run2.json
python bioasq_format_converter.py Batch01/runs/ranx_run3.json
python bioasq_format_converter.py Batch01/runs/ranx_run4.json