# CrossEntityQA config for ircot_qa + flan-t5-xxl
local dataset = "crossentityqa";
local retrieval_corpus_name = dataset;
local add_pinned_paras = false;
local valid_qids = null;
local prompt_reader_args = {
    "order_by_key": "qid",
    "estimated_generation_length": 0,
    "shuffle": false,
    "model_length_limit": 1000000,
    "tokenizer_model_name": "google/flan-t5-xxl",
};

local llm_retrieval_count = null;
local llm_map_count = null;
local bm25_retrieval_count = 6;
local rc_context_type_ = "gold_with_n_distractors";
local distractor_count = "2";
local rc_context_type = (
    if rc_context_type_ == "gold_with_n_distractors"
    then "gold_with_" + distractor_count + "_distractors"  else rc_context_type_
);
local multi_step_show_titles = null;
local multi_step_show_paras = null;
local multi_step_show_cot = null;
local rc_qa_type = "direct";
local qa_question_prefix = (
    if std.endsWith(rc_context_type, "cot")
    then "Answer the following question by reasoning step-by-step.\n"
    else "Answer the following question.\n"
);

{
    "start_state": "step_by_step_bm25_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_bm25_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "return_pids": false,
            "cumulate_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "cot_reasoning_gen",
            "next_model": "[EOQ]",
            "qa_type": rc_qa_type,
            "context_type": rc_context_type,
            "question_prefix": qa_question_prefix,
            "multi_step_show_titles": multi_step_show_titles,
            "multi_step_show_paras": multi_step_show_paras,
            "multi_step_show_cot": multi_step_show_cot,
            "end_state": "[EOQ]",
            "answer_is_numbered_list": true,
        },
    },
    "prompt_reader_args": prompt_reader_args,
    "dataset": dataset,
    "retrieval_corpus_name": retrieval_corpus_name,
    "add_pinned_paras": add_pinned_paras,
}