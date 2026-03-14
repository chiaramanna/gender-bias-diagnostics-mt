import argparse
import pickle
import numpy as np
import logging
import re

from transformers import AutoTokenizer, AutoProcessor, LlamaForCausalLM


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


def get_subword_indices(tokens, word, start_index=0, tgt_lang=None):
    word = word.lower()
    tokens = [token.lower() for token in tokens]
    word_parts = word.split()

    ARTICLE_SETS = {
        "it": {
            "la", "lo", "il",
            "uno", "un", "una",
            "della", "dello",
            "alla", "allo",
            "al",
        },
        "de": {
            "der", "die", "das",
            "den", "dem", "des",
            "ein", "eine", "einen", "einem", "eines",
            "am", "ans", "beim",
            "vom", "zum", "zur",
        },
        "es": {
            "el", "la",
            "un", "una",
            "al", "del",
        },
        "fr": {
            "le", "la",
            "un", "une",
            "du",
            "au", "aux",
        },
    }

    article_tokens = ARTICLE_SETS.get(tgt_lang, set())

    APOSTROPHE_PREV = {
        "it": {"un", "l", "dell", "all"},
        "fr": {"l"},
    }

    apostrophe_prev_tokens = APOSTROPHE_PREV.get(tgt_lang, set())

    indices = []
    i = start_index
    for part in word_parts:
        part_indices = []
        while i < len(tokens):
            token = tokens[i]

            # skip standalone '▁' tokens
            if token == "▁":
                i += 1
                continue
            
            # skip gendered words unless clean boundary (sheriff, etc...)
            if part in ["she", "he", "her", "him", "his"]:
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if not (
                        next_token.startswith("▁")
                        or next_token in [".", ",", "!", "?", ":", ";", "-", "(", ")", '"']
                    ):
                        i += 1
                        continue

            if token.startswith("▁"):
                subword = token[1:]
                if part.startswith(subword):
                    remaining_part = part[len(subword):]
                    matched_tokens = [token]
                    d = i

                    while d + 1 < len(tokens) and remaining_part:
                        next_token = tokens[d + 1]
                        if remaining_part.startswith(next_token):
                            matched_tokens.append(next_token)
                            remaining_part = remaining_part[len(next_token):]
                        else:
                            break
                        d += 1

                    if not remaining_part:
                        # ---- ADD ARTICLES / CLITICS HERE (only after a successful match)
                        if i - 1 >= 0 and tokens[i - 1].lstrip("▁").lower() in article_tokens:
                            indices.append(i - 1)

                        # include patterns like: [▁l, ', ...] or [▁un, ', ...]
                        if i - 2 >= 0 and tokens[i - 1] == "'" and tokens[i - 2].lstrip("▁").lower() in apostrophe_prev_tokens:
                            indices.append(i - 2)
                            indices.append(i - 1)

                        for j in range(i, d + 1):
                            indices.append(j)

                    part = part[len(subword):]

            # token does not start with '▁'
            elif part.startswith(token):
                remaining_part = part[len(token):]
                matched_tokens = [token]
                d = i

                while d + 1 < len(tokens) and remaining_part:
                    next_token = tokens[d + 1]
                    if remaining_part.startswith(next_token):
                        remaining_part = remaining_part[len(next_token):]
                    else:
                        break
                    d += 1

                if not remaining_part:
                    if i - 1 >= 0 and tokens[i - 1].lstrip("▁").lower() in article_tokens:
                        indices.append(i - 1)

                    if i - 2 >= 0 and tokens[i - 1] == "'" and tokens[i - 2].lstrip("▁").lower() in apostrophe_prev_tokens:
                        indices.append(i - 2)
                        indices.append(i - 1)

                    for j in range(i, d + 1):
                        indices.append(j)

            i += 1
            if not part:
                break

        indices.extend(part_indices)
        indices = list(set(indices))

    indices.sort()
    return indices


def find_target_word_index(target_word, sentence):
    input_words = sentence.split()
    target_word = target_word.lower()

    for i, word in enumerate(input_words):
        if word.lower().strip(',.') == target_word:
            return i


def get_target_output_token_indices(
    alignment_file,
    line_num,
    target_word_index,
    translated_sentence,
    translated_tokens,
    start_index=0,
    tgt_lang=None,
):
    with open(alignment_file, "r", encoding="utf-8") as file:
        alignments = file.readlines()

    alignment = alignments[line_num].strip()

    target_output_index = [
        int(tgt_idx)
        for src_idx, tgt_idx in (pair.split("-") for pair in alignment.split())
        if int(src_idx) == target_word_index
    ]

    if not target_output_index:
        print("Target word index not found in alignment file.")
        return []

    translated_words = translated_sentence.split()
    target_words = [translated_words[tgt_idx] for tgt_idx in target_output_index]

    target_word_subword_indices = []
    for word in target_words:
        if word[-1] == ".":
            word = word[:-1]
        elif word[-1] == ",":
            word = word[:-1]

        target_word_subword_indices.extend(
            get_subword_indices(
                translated_tokens,
                word,
                start_index=start_index,
                tgt_lang=tgt_lang,
            )
        )

    x = list(set(target_word_subword_indices))
    x.sort()
    return x


def build_prompt(tokenizer, sentence, model_name, tgt_lang):
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


def process_sentence(
    input_file,
    translation_file,
    results,
    tokenizer,
    output_file_attention,
    alignment_file,
    model_name,
    tgt_lang,
):
    attention_scores = []

    with open(results, "rb") as f:
        att_scores = pickle.load(f)

    num_layers = att_scores[0].shape[0]
    num_heads = att_scores[0].shape[1]

    with open(input_file, "r", encoding="utf-8") as infile, open(translation_file, "r", encoding="utf-8") as tfile:
        for line_num, (line, tline) in enumerate(zip(infile, tfile)):
            cols = line.strip().split("\t")
            tcols = tline.strip().split(" ||| ")

            input_sentence = cols[2]
            translated_sentence = tcols[1].strip()
            target_word = cols[3]
            gender = cols[0]

            logging.info(
                f"Input: {input_sentence}, Translated: {translated_sentence}, Target_word: {target_word}"
            )

            if gender == "neutral":
                attention_scores.append({
                    "sentence_num": line_num,
                    "target_word": target_word,
                    "cue_word": None,
                    "scores": np.ones((num_layers, num_heads)).tolist(),
                    "max_scores": np.ones((num_layers, num_heads)).tolist(),
                })
                continue

            if translated_sentence == "":
                attention_scores.append({
                    "sentence_num": line_num,
                    "target_word": target_word,
                    "cue_word": None,
                    "scores": np.ones((num_layers, num_heads)).tolist(),
                    "max_scores": np.ones((num_layers, num_heads)).tolist(),
                })
                continue

            if len(target_word.split()) > 1:
                attention_scores.append({
                    "sentence_num": line_num,
                    "target_word": target_word,
                    "cue_word": None,
                    "scores": np.ones((num_layers, num_heads)).tolist(),
                    "max_scores": np.ones((num_layers, num_heads)).tolist(),
                })
                continue

            cue_words = {
                "female": ["she", "her"],
                "male": ["he", "him", "his"],
            }

            clean_sentence = re.sub(r"[^\w\s]", "", input_sentence.lower())
            cue_word = next(
                (w for w in cue_words[gender.lower()] if w in clean_sentence.split()),
                None,
            )

            if cue_word is None:
                print(f"No cue word found for sentence {line_num}. Skipping.")
                attention_scores.append({
                    "sentence_num": line_num,
                    "target_word": target_word,
                    "cue_word": None,
                    "scores": np.ones((num_layers, num_heads)).tolist(),
                    "max_scores": np.ones((num_layers, num_heads)).tolist(),
                })
                continue

            logging.info(f"Cue: {cue_word}")

            prompt = build_prompt(tokenizer, input_sentence, model_name, tgt_lang=tgt_lang)
            full_context = prompt + " " + translated_sentence

            cfg = MODEL_CONFIG[model_name]
            encoded = tokenizer(
                full_context,
                return_tensors="pt",
                add_special_tokens=not cfg.get("chat", False),
            )
            full_context_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

            cue_input_indices = get_subword_indices(full_context_tokens, cue_word)
            target_word_index = find_target_word_index(target_word, input_sentence)
            target_input_indices = get_subword_indices(full_context_tokens, target_word)

            prompt_encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=not cfg.get("chat", False),
            )
            prompt_length = prompt_encoded["input_ids"].shape[1]

            target_output_indices = get_target_output_token_indices(
                alignment_file,
                line_num,
                target_word_index,
                translated_sentence,
                full_context_tokens,
                start_index=prompt_length,
                tgt_lang=tgt_lang,
            )

            target_output_indices = [idx - 1 for idx in target_output_indices]

            print(target_output_indices)

            if not target_output_indices:
                attention_scores.append({
                    "sentence_num": line_num,
                    "target_word": target_word,
                    "cue_word": cue_word,
                    "scores": np.ones((num_layers, num_heads)).tolist(),
                    "max_scores": np.ones((num_layers, num_heads)).tolist(),
                })
                continue

            attention_matrix = att_scores[line_num]

            sentence_attention_scores = []
            sentence_attention_max_cue_scores = []

            print(line_num)
            for layer_idx in range(attention_matrix.shape[0]):
                layer_scores = []
                layer_max_scores = []

                for head_idx in range(attention_matrix.shape[1]):
                    target_token_attention_scores = []
                    for t_idx in target_output_indices:
                        score = attention_matrix[
                            layer_idx,
                            head_idx,
                            t_idx,
                            cue_input_indices[0]
                        ]
                        target_token_attention_scores.append(score)

                    target_token_attention_scores = np.squeeze(target_token_attention_scores)

                    layer_scores.append(np.nanmean(target_token_attention_scores))
                    layer_max_scores.append(np.nanmax(target_token_attention_scores))

                sentence_attention_scores.append(layer_scores)
                sentence_attention_max_cue_scores.append(layer_max_scores)

            attention_scores.append({
                "sentence_num": line_num,
                "target_word": target_word,
                "cue_word": cue_word,
                "scores": sentence_attention_scores,
                "max_scores": sentence_attention_max_cue_scores,
            })

    with open(output_file_attention, "wb") as f:
        pickle.dump(attention_scores, f)


def main():
    parser = argparse.ArgumentParser(description="Extract attention scores.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--translation_file", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output_file_attention", type=str, required=True)
    parser.add_argument("--alignment_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--tgt_lang", type=str, required=True, choices=list(LANG_NAMES.keys()))

    args = parser.parse_args()

    log_filename = f"./logs/{args.model_name}/extract_attention_{args.tgt_lang}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg = MODEL_CONFIG[args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg["id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    process_sentence(
        args.input_file,
        args.translation_file,
        args.results,
        tokenizer,
        args.output_file_attention,
        args.alignment_file,
        args.model_name,
        args.tgt_lang,
    )


if __name__ == "__main__":
    main()