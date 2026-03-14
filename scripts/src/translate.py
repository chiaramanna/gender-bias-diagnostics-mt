import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch
import os
import logging


### MODEL DICTIONARY ###

MODEL_CONFIG = {
    "Llama2": {"id": "meta-llama/Llama-2-7b-hf", "chat": False},
    "TowerBase": {"id": "Unbabel/TowerBase-7B-v0.1", "chat": False},
    "TowerInstruct-v0.1": {"id": "Unbabel/TowerInstruct-7B-v0.1", "chat": True},
    "TowerInstruct-v0.2": {"id": "Unbabel/TowerInstruct-7B-v0.2", "chat": True},
}

LANG_NAMES = {
    "en": "English",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


### TRANSLATION PIPELINE ###

def load_translation_pipeline(model_name: str, quantization: bool = True):
    """
    Load the model and tokenizer based on the model name, with optional quantization.
    This function is called to avoid reloading the model every time.
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Available: {', '.join(MODEL_CONFIG.keys())}"
        )

    cfg = MODEL_CONFIG[model_name]
    model_id = cfg["id"]

    quantization_config = (
        BitsAndBytesConfig(load_in_4bit=True) if quantization else None
    )

    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def build_prompt(
    tokenizer,
    sentence: str,
    model_name: str,
    src_lang: str = "en",
    tgt_lang: str = "it",
) -> str:
    cfg = MODEL_CONFIG[model_name]
    is_chat = cfg.get("chat", False)
    tgt_lang_name = LANG_NAMES[tgt_lang]

    user_content = (
            f"Translate the following text from English into {tgt_lang_name}.\n"
            f"English: {sentence}\n"
            f"{tgt_lang_name}:"
        )

    if not is_chat:
        return user_content

    else:
        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def translate_sentence(model, tokenizer, sentence, model_name, src_lang="en", tgt_lang="it"):
    """
    Translate a single sentence using the loaded model and tokenizer.
    The prompt changes depending on the model type.
    """

    cfg = MODEL_CONFIG[model_name]

    prompt = build_prompt(
        sentence=sentence,
        model_name=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tokenizer=tokenizer,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=not cfg.get("chat", False),
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    logging.info(generated_text)

    def extract_translation(generated_text, model_name, tgt_lang="it"):
        tgt_lang_name = LANG_NAMES[tgt_lang]
        if model_name in ["TowerInstruct-v0.1", "TowerInstruct-v0.2"]:
            start_index = generated_text.find(f"{tgt_lang_name}:")
            if start_index != -1:
                assistant_index = generated_text.find("assistant", start_index + len(f"{tgt_lang_name}:"))
                if assistant_index != -1:
                    translated_text = generated_text[assistant_index + len("assistant"):].strip()
                else:
                    translated_text = generated_text[start_index + len(f"{tgt_lang_name}:"):].strip()

                end_index = translated_text.find(".")
                if end_index != -1:
                    return translated_text[:end_index + 1].strip()
                else:
                    return translated_text.strip()
            else:
                return None

        else:
            start_index = generated_text.find(f"{tgt_lang_name}:")
            if start_index != -1:
                translated_text = generated_text[start_index + len(f"{tgt_lang_name}:"):].strip()
                end_index = translated_text.find(".")
                if end_index != -1:
                    return translated_text[:end_index + 1].strip()
            return None

    translated_text = extract_translation(generated_text, model_name, tgt_lang=tgt_lang)

    # retry
    if translated_text is None:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=not cfg.get("chat", False),
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        translated_text = extract_translation(generated_text, model_name, tgt_lang=tgt_lang)

        if translated_text is None:
            logging.error(
                f"Failed to extract translation after retry for sentence: '{sentence}', generated text: '{generated_text}'"
            )
            translated_text = ""

    return translated_text


def translate_and_save_log(input_file, output_file, model_name, src_lang, tgt_lang, quantization=True):
    """
    Translates sentences from an input file and saves the results in the specified format.
    """

    model, tokenizer = load_translation_pipeline(model_name, quantization=quantization)

    with open(input_file, 'r', encoding="utf-8") as infile, open(output_file, 'w', encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, start=1):
            sentence = line.strip().split('\t')[2]

            translated_text = translate_sentence(
                model,
                tokenizer,
                sentence,
                model_name,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            logging.info(f"Translating sentence {line_num}: {translated_text}")
            outfile.write(f"{sentence} ||| {translated_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Translate sentences using a specified model.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--model_name', type=str, required=True, help='Pre-trained model name or path')
    parser.add_argument(
        "--src_lang",
        type=str,
        default="en",
        choices=LANG_NAMES.keys(),
        help="Source language code",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="it",
        choices=LANG_NAMES.keys(),
        help="Target language code",
    )
    parser.add_argument('--quantization', action='store_true', help='Enable 4-bit quantization')

    args = parser.parse_args()

    log_dir = f'./logs/{args.model_name}'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f'./logs/{args.model_name}/translation_{args.tgt_lang}.log'
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    translate_and_save_log(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()