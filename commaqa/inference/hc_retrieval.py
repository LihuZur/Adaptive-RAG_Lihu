"""HC-based retrieval participant for Adaptive-RAG."""

import json
import logging
import numpy as np
from pathlib import Path

from commaqa.inference.model_search import ParticipantModel
from commaqa.inference.data_instances import QuestionAnsweringStep
from commaqa.inference.dataset_readers import get_pid_for_title_paragraph_text
from commaqa.inference.ircot import safe_post_request, remove_wh_words, is_para_closely_matching
from commaqa.hc import HigherCriticism, NullDistribution

logger = logging.getLogger(__name__)


class HCRetrieveAndSelectParticipant(ParticipantModel):
    """
    Higher Criticism-based adaptive retrieval participant.
    
    Retrieves candidate documents via BM25, then uses HC statistics
    to adaptively determine how many documents to actually use.
    """

    def __init__(
        self,
        retriever_host=None,
        retriever_port=None,
        retrieval_count=None,  # Initial candidates to retrieve
        gamma=0.1,  # HC search window parameter
        min_hc=0.0,  # Minimum HC statistic threshold
        null_dist_path=None,  # Path to pre-computed null distribution
        query_source="original_question",
        source_corpus_name=None,
        document_type="title_paragraph_text",
        global_max_num_paras=15,
        dont_skip_long_paras=False,
        return_pids=False,
        return_paras=False,
        next_model=None,
        end_state="[EOQ]",
    ):
        """
        Initialize HC retrieval participant.

        Args:
            retriever_host: Elasticsearch host
            retriever_port: Elasticsearch port
            retrieval_count: Number of candidate documents to retrieve (e.g., 30)
            gamma: HC search window (fraction of candidates to consider)
            min_hc: Minimum HC statistic to select any documents
            null_dist_path: Path to saved null distribution (.pkl file)
            query_source: Source of query ('original_question')
            source_corpus_name: Corpus name (e.g., 'hotpotqa')
            document_type: Type of documents to retrieve
            global_max_num_paras: Maximum total paragraphs
            dont_skip_long_paras: Whether to skip long paragraphs
            return_pids: Return paragraph IDs
            return_paras: Return full paragraphs
            next_model: Next model in pipeline
            end_state: End state marker
        """
        assert query_source in (
            "original_question",
            "last_answer",
            "question_or_last_generated_sentence",
        ), f"query_source {query_source} not valid"

        assert document_type in ("title", "paragraph_text", "title_paragraph_text")

        if retrieval_count is None:
            raise ValueError("retrieval_count is required (number of candidates to retrieve)")
        if source_corpus_name is None:
            raise ValueError("source_corpus_name is required")

        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.retrieval_count = retrieval_count
        self.gamma = gamma
        self.min_hc = min_hc
        self.query_source = query_source
        self.source_corpus_name = source_corpus_name
        self.document_type = document_type
        self.global_max_num_paras = global_max_num_paras
        self.dont_skip_long_paras = dont_skip_long_paras
        self.return_pids = return_pids
        self.return_paras = return_paras
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0

        if return_pids and return_paras:
            raise ValueError("Only one of return_pids or return_paras should be true")

        # Load null distribution
        self.null_distribution = None
        if null_dist_path:
            null_dist_path = Path(null_dist_path)
            if null_dist_path.exists():
                self.null_distribution = NullDistribution.load(str(null_dist_path))
                logger.info(f"Loaded null distribution from {null_dist_path}")
            else:
                logger.warning(f"Null distribution not found at {null_dist_path}, HC will not work!")

        # Initialize HC module
        self.hc = HigherCriticism(null_distribution=self.null_distribution)

        self.retrieval_failures_so_far = 0
        self.retrieval_failures_max = 9

    def return_model_calls(self):
        return {"hc_retrieve_and_select": self.num_calls}

    def query(self, state, debug=False):
        """
        Execute HC-based retrieval and selection.

        1. Retrieve candidate documents via BM25
        2. Extract BM25 scores
        3. Apply HC to determine optimal k
        4. Select top-k documents
        """

        # Determine query text
        if self.query_source == "original_question":
            input_query = state.data["question"]
        elif self.query_source == "last_answer":
            input_query = state.data.get_last_answer()
        elif self.query_source == "question_or_last_generated_sentence":
            question = state.data["question"]
            generated_sentences = state.data.get("generated_sentences", [])
            last_generated_sentence_str = generated_sentences[-1].strip() if generated_sentences else ""
            input_query = last_generated_sentence_str if last_generated_sentence_str else question
        else:
            raise ValueError(f"Unknown query_source: {self.query_source}")

        # Initialize results
        selected_titles = []
        selected_paras = []

        # Prepare BM25 retrieval
        input_query = remove_wh_words(input_query)

        if not input_query.strip():
            # Empty query - return empty results
            self.num_calls += 1
            answer = json.dumps(selected_titles)
            new_state = state.copy()
            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
            new_state.next = self.next_model if self.next_model else self.end_state
            new_state.data["paras"] = selected_paras
            new_state.data["titles"] = selected_titles
            return new_state

        params = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": input_query,
            "max_hits_count": self.retrieval_count,
            "corpus_name": self.source_corpus_name,
            "document_type": self.document_type,
        }

        url = self.retriever_host.rstrip("/") + ":" + str(self.retriever_port) + "/retrieve"
        result = safe_post_request(url, params)

        if not result.ok:
            self.retrieval_failures_so_far += 1
            if self.retrieval_failures_so_far > self.retrieval_failures_max:
                raise Exception(
                    f"Retrieval failure exceeded max allowed times "
                    f"({self.retrieval_failures_so_far} > {self.retrieval_failures_max})"
                )
            logger.warning(
                f"Retrieval failed {self.retrieval_failures_so_far} times. Returning empty result."
            )
            self.num_calls += 1
            answer = json.dumps(selected_titles)
            new_state = state.copy()
            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
            new_state.next = self.next_model if self.next_model else self.end_state
            new_state.data["paras"] = selected_paras
            new_state.data["titles"] = selected_titles
            return new_state

        # Parse retrieval results
        result = result.json()
        retrieval = result["retrieval"]

        if not retrieval:
            # No results from BM25
            logger.debug("No retrieval results, returning empty set")
            self.num_calls += 1
            answer = json.dumps(selected_titles)
            new_state = state.copy()
            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
            new_state.next = self.next_model if self.next_model else self.end_state
            new_state.data["paras"] = selected_paras
            new_state.data["titles"] = selected_titles
            return new_state

        # Extract BM25 scores and documents
        candidate_scores = []
        candidate_docs = []

        for retrieval_item in retrieval:
            if retrieval_item["corpus_name"] != self.source_corpus_name:
                logger.warning(
                    f"Retrieved corpus {retrieval_item['corpus_name']} != {self.source_corpus_name}"
                )
                continue

            # Skip excessively long paragraphs
            if (len(retrieval_item["paragraph_text"].split(" ")) > 600 and
                not self.dont_skip_long_paras):
                continue

            score = retrieval_item.get("score", 0.0)
            candidate_scores.append(score)
            candidate_docs.append(retrieval_item)

        # Apply HC if we have a null distribution
        if self.null_distribution is not None and len(candidate_scores) > 0:
            scores_array = np.array(candidate_scores, dtype=np.float32)

            # Compute HC threshold
            hc_result = self.hc.compute_hc_threshold(
                scores_array,
                gamma=self.gamma,
                min_hc=self.min_hc,
                allow_empty=True
            )

            k_selected = hc_result.k

            if debug:
                logger.info(f"HC: k={k_selected}/{len(candidate_scores)}, "
                          f"HC stat={hc_result.hc_statistic:.3f}, "
                          f"threshold={hc_result.threshold:.3f}")
        else:
            # Fallback: use all candidates if no null distribution
            k_selected = len(candidate_scores)
            if debug:
                logger.warning("No null distribution, using all candidates")

        # Select top-k documents
        # Sort by score descending
        sorted_indices = np.argsort(candidate_scores)[::-1]

        for i in range(min(k_selected, len(sorted_indices))):
            idx = sorted_indices[i]
            retrieval_item = candidate_docs[idx]

            # Check for duplicates
            if is_para_closely_matching(
                selected_titles,
                selected_paras,
                retrieval_item["title"],
                retrieval_item["paragraph_text"],
            ):
                continue

            # Check global limit
            if len(selected_paras) >= self.global_max_num_paras:
                break

            selected_titles.append(retrieval_item["title"])
            selected_paras.append(retrieval_item["paragraph_text"])

        self.num_calls += 1

        # Format answer
        answer = json.dumps(selected_titles)

        if self.return_pids:
            pids = [
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(selected_titles, selected_paras)
            ]
            answer = json.dumps(pids)

        if self.return_paras:
            answer = json.dumps(
                [{"title": title, "paragraph_text": para}
                 for title, para in zip(selected_titles, selected_paras)]
            )

        # Create new state
        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state
        new_state.data["paras"] = selected_paras
        new_state.data["titles"] = selected_titles

        return new_state
