import os
os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(os.getcwd()) + '/cache'

import time
from functools import lru_cache

from fastapi import FastAPI
import torch

from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)


##########################################
# FORCE CPU ALWAYS
##########################################
DEVICE = torch.device("cpu")


@lru_cache(maxsize=None)
def get_model_and_tokenizer():
    model_shortname = os.environ["MODEL_NAME"]

    valid_model_shortnames = [
        "gpt-j-6B",
        "opt-66b",
        "gpt-neox-20b",
        "T0pp",
        "opt-125m",
        "flan-t5-base",
        "flan-t5-large",
        "flan-t5-xl",
        "flan-t5-xxl",
        "flan-t5-base",
    ]
    assert model_shortname in valid_model_shortnames, f"Invalid model: {model_shortname}"

    #
    # CPU-SAFE LOADING (NO device_map, NO offload_folder)
    #
    if model_shortname == "gpt-j-6B":
        model_name = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="sharded").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-66b":
        model_name = "facebook/opt-66b"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname == "gpt-neox-20b":
        model_name = "EleutherAI/gpt-neox-20b"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "T0pp":
        model_name = "bigscience/T0pp"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="sharded").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_shortname == "opt-125m":
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, revision="main").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    elif model_shortname.startswith("flan-t5"):
        model_name = "google/" + model_shortname
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main").to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        raise ValueError("Model not supported in CPU mode")

    return model, tokenizer


class EOSReachedCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, eos_text: str):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert len(self.tokenizer.encode(eos_text)) < 10

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        return decoded_text.strip().endswith(self.eos_text.strip())


app = FastAPI()


@app.get("/")
async def index():
    model_shortname = os.environ["MODEL_NAME"]
    return {"message": f"Server running for {model_shortname}. Use /generate/."}


@app.get("/generate/")
async def generate(
    prompt: str,
    max_input: int = None,
    max_length: int = 200,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: float = None,
    length_penalty: float = None,
    eos_text: str = None,
    keep_prompt: bool = False,
):

    start_time = time.time()

    model_shortname = os.environ["MODEL_NAME"]
    model, tokenizer = get_model_and_tokenizer()

    # CPU input
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=max_input).to(DEVICE)

    stopping_criteria_list = StoppingCriteriaList()
    if eos_text:
        stopping_criteria_list = StoppingCriteriaList([EOSReachedCriteria(tokenizer, eos_text)])

    is_encoder_decoder = model_shortname in ["T0pp"] or model_shortname.startswith("flan-t5")

    generated_output = model.generate(
        inputs,
        max_new_tokens=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        stopping_criteria=stopping_criteria_list,
        output_scores=False,
    )

    generated_ids = generated_output["sequences"]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if not keep_prompt and not is_encoder_decoder:
        generated_texts = [txt[len(prompt):] if txt.startswith(prompt) else txt for txt in generated_texts]
    elif keep_prompt and is_encoder_decoder:
        generated_texts = [prompt + txt for txt in generated_texts]

    end_time = time.time()

    return {
        "generated_num_tokens": [len(x) for x in generated_ids],
        "generated_texts": generated_texts,
        "run_time_in_seconds": end_time - start_time,
        "model_name": model_shortname,
    }


print("\nLoading model and tokenizer on CPU...")
get_model_and_tokenizer()
print("Loaded.\n")
