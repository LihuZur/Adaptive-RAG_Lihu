local dataset = "crossentityqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = false;
local valid_qids = std.extVar("valid_qids");
local prompt_reader_args = {
  "order_by_key": "qid",
  "estimated_generation_length": 0,
  "shuffle": false,
  "model_length_limit": 1000000,
  "tokenizer_model_name": "google/flan-t5-xl",
};

# (Potentially) Hyper-parameters:
local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 5;
local rc_context_type_ = "gold_with_n_distractors";
local distractor_count = "2";
local rc_context_type = (
  if rc_context_type_ == "gold_with_n_distractors"
  then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local rc_qa_type = "cot";

{
  "start_state": "generate_titles",
  "end_state": "[EOQ]",
  "models": {
    "generate_titles": {
      "name": "retrieve_and_reset_paragraphs",
      "next_model": "generate_main_question",
      "retrieval_type": "bm25",
      "retriever_host": std.extVar("RETRIEVER_HOST"),
      "retriever_port": std.extVar("RETRIEVER_PORT"),
      "retrieval_count": bm25_retrieval_count,
      "global_max_num_paras": 15,
      "query_source": "original_question",
      "source_corpus_name": retrieval_corpus_name,
      "document_type": "title_paragraph_text",
      "end_state": "[EOQ]",
    },
    "generate_main_question": {
      "name": "copy_question",
      "next_model": "answer_main_question",
      "eoq_after_n_calls": 1,
      "end_state": "[EOQ]",
    },
    "answer_main_question": {
      "name": "llmqa",
      "next_model": if std.endsWith(rc_qa_type, "cot") then "extract_answer" else null,
      "prompt_file": "prompts/"+dataset+"/"+rc_context_type+"_context_"+rc_qa_type+"_qa_codex.txt",
      "prompt_reader_args": prompt_reader_args,
      "valid_qids": valid_qids,
      "end_state": "[EOQ]",
      "gen_model": "gpt3",
      "engine": "gpt-4o-mini",
      "retry_after_n_seconds": 50,
      "add_context": true,
    },
    "extract_answer": {
      "name": "answer_extractor",
      "query_source": "last_answer",
      "regex": ".* answer is:? (.*)\\.?",
      "match_all_on_failure": true,
      "remove_last_fullstop": true,
    }
  },
  "reader": {
    "name": "multi_para_rc",
    "add_paras": false,
    "add_gold_paras": false,
    "add_pinned_paras": add_pinned_paras,
  },
  "prediction_type": "answer",
  "dataset": dataset,
  "retrieval_corpus_name": retrieval_corpus_name,
  "add_pinned_paras": add_pinned_paras,
}
