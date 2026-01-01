import logging
import time
import os
from functools import lru_cache

import openai
from diskcache import Cache
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit


logger = logging.getLogger(__name__)


cache = Cache(os.path.expanduser("~/.cache/gpt3calls"))


@cache.memoize()
def cached_openai_call(  # kwargs doesn't work with caching.
    prompt,
    engine,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    best_of,
    logprobs,
):
    # Use ChatCompletion API for chat models (gpt-4, gpt-4o, gpt-3.5-turbo variants except instruct)
    is_chat_model = any(model in engine for model in ["gpt-4", "gpt-3.5-turbo"]) and "instruct" not in engine

    if is_chat_model:
        return openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
        )
    else:
        return openai.Completion.create(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            best_of=best_of,
            logprobs=logprobs,
        )


def openai_call(
    prompt,
    engine,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    best_of,
    logprobs,
):
    # Use ChatCompletion API for chat models
    is_chat_model = any(model in engine for model in ["gpt-4", "gpt-3.5-turbo"]) and "instruct" not in engine

    if temperature == 0:
        function = cached_openai_call
        return function(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            best_of=best_of,
            logprobs=logprobs,
        )
    elif is_chat_model:
        return openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
        )
    else:
        return openai.Completion.create(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            best_of=best_of,
            logprobs=logprobs,
        )


@lru_cache(maxsize=1)
def get_gpt_tokenizer():
    """Get tokenizer for counting GPT tokens"""
    try:
        import tiktoken
        # Use tiktoken for accurate GPT token counting
        encoding = tiktoken.get_encoding("cl100k_base")

        # Create a minimal wrapper with tokenize method
        class TiktokenWrapper:
            def __init__(self, enc):
                self.enc = enc
            def tokenize(self, text):
                return self.enc.encode(text)

        return TiktokenWrapper(encoding)
    except ImportError:
        # Fallback to GPT-2 tokenizer if tiktoken not available
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained("gpt2")


class GPT3Generator:
    def __init__(
        self,
        engine="gpt-3.5-turbo-instruct",
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
        retry_after_n_seconds=None,
        n=1,
        best_of=1,
        logprobs=0,
        remove_method="first",
    ):
        self.engine = engine
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        self.retry_after_n_seconds = retry_after_n_seconds
        self.remove_method = remove_method

        # if "code-davinci" not in engine:
        #     raise Exception("Not allowed to prevent accidental $$ wastage.")

        # if "code-davinci" not in engine and self.retry_after_n_seconds is not None:
        #     raise Exception(
        #         "Retry is only supported for code-davinci as it's free. "
        #         "Using it for other paid models is risky and so is disabled."
        #     )

        # Set context limits based on model
        if "code-davinci" in engine:
            self.model_tokens_limit = 8000
        elif "gpt-4o" in engine:
            self.model_tokens_limit = 120000  # GPT-4o/4o-mini: 128K context
        elif "gpt-4" in engine:
            self.model_tokens_limit = 120000  # GPT-4: 128K context (for turbo variants)
        else:
            self.model_tokens_limit = 3500  # GPT-3.5 and others

    def generate_text_sequence(self, prompt):
        prompt_trunc = prompt[:120].replace('\n', ' ')
        # Force logger to INFO level and add StreamHandler if not present
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.debug(f"[EMBEDDING] Starting embedding for prompt (truncated): {prompt_trunc} ...")
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        # GPT3 can't handle trailing white-space
        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name="gpt2",  # did this before tiktoken was released.
            last_is_test_example=True,
        )
        #import pdb; pdb.set_trace()

        arguments = {
            "engine": self.engine,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "best_of": self.best_of,
            "logprobs": self.logprobs,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }
        if self.best_of is not None:
            arguments["best_of"] = self.best_of

        success = False

        for index in range(500):
            try:
                logger.debug(f"[EMBEDDING] Attempt {index+1} for embedding.")
                response = openai_call(**arguments)
                success = True
                logger.debug(f"[EMBEDDING] Embedding call succeeded on attempt {index+1}.")
                break
            except Exception as exception:
                success = False
                logger.error(f"[EMBEDDING] Exception during embedding attempt {index+1}: {exception}")
                tokenizer = get_gpt_tokenizer()
                prompt_num_tokens = len(tokenizer.tokenize(prompt))
                if prompt_num_tokens + arguments["max_tokens"] > self.model_tokens_limit > prompt_num_tokens:
                    last_used_max_tokens = arguments["max_tokens"]
                    updated_max_tokens = self.model_tokens_limit - prompt_num_tokens
                    arguments["max_tokens"] = updated_max_tokens
                    if last_used_max_tokens == updated_max_tokens:
                        logger.error(f"[EMBEDDING] Could not reduce max_tokens further. Failing embedding.")
                        break
                    logger.warning(
                        f"[EMBEDDING] (Round {index}) Decreasing max_tokens from {last_used_max_tokens} to {updated_max_tokens} and retrying."
                    )
                    continue

                if self.retry_after_n_seconds is None:
                    import traceback
                    logger.error(traceback.format_exc())
                    raise

                logger.warning(f"[EMBEDDING] Potentially reached OpenAI rate limit. Will try again in {self.retry_after_n_seconds}s.")
                time.sleep(self.retry_after_n_seconds)
                pass

        if not success:
            logger.error("[EMBEDDING] Could not complete OpenAI call after 500 attempts.")
            raise Exception("Could not complete OpenAI call")

        # Print only the interesting part of the LLM response (the main content)
        interesting_content = None
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                interesting_content = choice["message"]["content"]
            elif "text" in choice:
                interesting_content = choice["text"]
        logger.debug(f"[LLM RESPONSE CONTENT] {interesting_content}")

        output_seq_score = []

        # Check if this is a ChatCompletion response
        is_chat_response = "message" in response["choices"][0] if response["choices"] else False

        for index, choice in enumerate(response["choices"]):
            # Extract text based on response type
            if is_chat_response:
                text = choice["message"]["content"]
                output_seq_score.append((text, index))
            elif "logprobs" in choice and "token_logprobs" in choice["logprobs"]:
                probs = []
                for prob, tok in zip(choice["logprobs"]["token_logprobs"], choice["logprobs"]["tokens"]):
                    if tok not in self.stop and tok != "<|endoftext|>":
                        probs.append(prob)
                    else:
                        probs.append(prob)
                        break

                score = -sum(probs) / len(probs) if len(probs) else 100.0
                output_seq_score.append((choice["text"], score))
            else:
                output_seq_score.append((choice["text"], index))

        output_trunc = output_seq_score[0][0][:120].replace('\n', ' ') if output_seq_score else ''
        logger.debug(f"[EMBEDDING] Embedding complete. Output: {output_trunc} ...")
        logger.debug(f"[EMBEDDING] Successfully finished embedding for prompt.")
        return sorted(output_seq_score, key=lambda x: x[1])

