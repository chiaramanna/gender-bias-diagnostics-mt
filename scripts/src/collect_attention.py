import os
import json
import logging
import torch
import numpy as np
import argparse
import pickle

from transformers import AutoTokenizer, AutoConfig, AutoProcessor, LlamaForCausalLM


MODEL_CONFIG = {
    "Llama2": {"id": "meta-llama/Llama-2-7b-hf", "chat": False, "family": "llama"},
    "TowerBase": {"id": "Unbabel/TowerBase-7B-v0.1", "chat": False, "family": "llama"},
    "TowerInstruct-v0.1": {"id": "Unbabel/TowerInstruct-7B-v0.1", "chat": True, "family": "llama"},
    "TowerInstruct-v0.2": {"id": "Unbabel/TowerInstruct-7B-v0.2", "chat": True, "family": "llama"},

}


LANG_NAMES = {
    "en": "English",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def build_prompt(tokenizer, sentence, model_name, tgt_lang):

    cfg = MODEL_CONFIG[model_name]
    tgt_lang_name = LANG_NAMES[tgt_lang]

    user_content = (
            f"Translate the following text from English into {tgt_lang_name}.\n"
            f"English: {sentence}\n"
            f"{tgt_lang_name}:"
        )

    if not cfg["chat"]:
        return user_content

    else:
        messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def process_file(input_file, translation_file, tokenizer, model_name, model, tgt_lang):

    attention_results = []

    with open(input_file) as infile, open(translation_file) as tfile:

        for line_num, (line, tline) in enumerate(zip(infile, tfile)):

            cols = line.strip().split("\t")
            tcols = tline.strip().split(" ||| ")

            input_sentence = cols[2]
            translated_sentence = tcols[1].strip()

            prompt = build_prompt(
                tokenizer,
                input_sentence,
                model_name,
                tgt_lang,
            )

            full_context = prompt + " " + translated_sentence

            logging.info(f"Processing sentence {line_num}")

            cfg = MODEL_CONFIG[model_name]

            inputs = tokenizer(
                full_context,
                return_tensors="pt",
                padding=True,
                add_special_tokens=not cfg.get("chat", False),
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_attentions=True,
                )

            attentions = outputs.attentions

            attention_scores = (
                torch.stack(attentions)
                .permute(1, 0, 2, 3, 4)
                .squeeze(0)
                .detach()
                .to(torch.float32)
                .cpu()
                .numpy()
            )

            attention_results.append(attention_scores)

    return attention_results


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--translation_file", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--model_name",
        required=True,
        choices=list(MODEL_CONFIG.keys()),
    )

    parser.add_argument(
        "--tgt_lang",
        default="it",
        choices=list(LANG_NAMES.keys()),
    )

    parser.add_argument("--suffix", default="")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"./logs/{args.model_name}", exist_ok=True)

    log_filename = (
        f"./logs/{args.model_name}/attention_{args.tgt_lang}_{args.suffix}.log"
    )

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg = MODEL_CONFIG[args.model_name]
    model_id = cfg["id"]
    target_lang = LANG_NAMES[args.tgt_lang]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    config._attn_implementation = "eager"

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        device_map="balanced",
        config=config,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    attention_results = process_file(
        args.input_file,
        args.translation_file,
        tokenizer,
        args.model_name,
        model,
        args.tgt_lang,
    )

    output_path = os.path.join(
        args.output_dir,
        f"final_results_{args.suffix}_{target_lang}.pkl",
    )

    with open(output_path, "wb") as f:
        pickle.dump(attention_results, f)

    logging.info(f"Finished. Saved results to {output_path}")


if __name__ == "__main__":
    main()