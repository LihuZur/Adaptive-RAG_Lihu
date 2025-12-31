
local dataset = "crossentityqa";
local retriever_host = std.extVar("RETRIEVER_HOST");
local retriever_port = std.extVar("RETRIEVER_PORT");
local llm_server_host = std.extVar("LLM_SERVER_HOST");
local llm_server_port = std.extVar("LLM_SERVER_PORT");
local corpus_name = dataset;
local dataset_name = dataset;
local valid_qids = null;
local prompt_reader_args = {
    "order_by_key": "qid",
    "estimated_generation_length": 300,
    "shuffle": false,
    "model_length_limit": 8000,
};

local bm25_retrieval_count = std.extVar("bm25_retrieval_count");
local gamma = std.parseJson(std.extVar("gamma"));
local min_hc = std.parseJson(std.extVar("min_hc"));
local null_dist_path = "processed_data/hc_null_distributions/" + corpus_name + "_null_dist.pkl";

{
  "start_state": "copy_question",
  "end_state": "[EOQ]",
  "models": {
    "copy_question": {
      "name": "copy_question",
      "next_model": "hc_retrieve_and_select",
      "eoq_after_n_calls": 1,
      "end_state": "[EOQ]",
    },
    "hc_retrieve_and_select": {
      "name": "hc_retrieve_and_select",
      "retriever_host": retriever_host,
      "retriever_port": retriever_port,
      "retrieval_count": bm25_retrieval_count,
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
      "prompt_file": "prompts/crossentityqa/no_context_cot_qa_codex.txt",
      "prompt_reader_args": prompt_reader_args,
      "gen_model": "gpt3",
      "engine": "gpt-4o-mini",
      "max_tokens": 400,
      "retry_after_n_seconds": 50,
      "add_context": true,
      "next_model": "[EOQ]",
      "answer_is_numbered_list": true,
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
