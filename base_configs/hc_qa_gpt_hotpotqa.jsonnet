local retriever_host = std.extVar("RETRIEVER_HOST");
local retriever_port = std.extVar("RETRIEVER_PORT");
local llm_server_host = std.extVar("LLM_SERVER_HOST");
local llm_server_port = std.extVar("LLM_SERVER_PORT");
local corpus_name = "hotpotqa";
local dataset_name = "hotpotqa";

local bm25_retrieval_count = std.extVar("bm25_retrieval_count");
local gamma = std.parseJson(std.extVar("gamma"));
local min_hc = std.parseJson(std.extVar("min_hc"));
local null_dist_path = "processed_data/hc_null_distributions/" + corpus_name + "_null_dist.pkl";

{
  "start_state": "hc_retrieve_and_select",
  "end_state": "[EOQ]",
  "models": {
    "hc_retrieve_and_select": {
      "name": "hc_retrieve_and_select",
      "retriever_host": retriever_host,
      "retriever_port": retriever_port,
      "retrieval_count": std.parseInt(bm25_retrieval_count),
      "gamma": gamma,
      "min_hc": min_hc,
      "null_dist_path": null_dist_path,
      "source_corpus_name": corpus_name,
      "document_type": "title_paragraph_text",
      "query_source": "original_question",
      "global_max_num_paras": 15,
      "return_pids": false,
      "return_paras": false,
      "next_model": "llmqa",
    },
    "llmqa": {
      "name": "llmqa",
      "max_length": 400,
      "lm_server_host": llm_server_host,
      "lm_server_port": llm_server_port,
      "lm_server_model_name": "gpt-4o-mini",
      "prompt_file": "prompts/hotpotqa/nor.txt",
      "disable_qm": true,
      "next_model": "[EOQ]",
    },
  },
  "reader": {
    "name": "multi_para_rc",
    "add_paras": false,
    "add_gold_paras": false,
    "add_pinned_paras": false,
  },
  "prediction_type": "answer",
}
