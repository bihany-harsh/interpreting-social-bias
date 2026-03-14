# TODO: to add multi-gpu support
# TODO: model the probability of the sentence and not the final token only (provide the abiluty to run both version)

# ** ACKNOWLEDGMENT **: 
# 1. a bug-report was generated after the code was written using Claude Opus 4.6, and then the bugs detected were fixed. (bug report to found within the ai-acknowledgment folder)

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import json
import jsonlines
import logging
from datetime import datetime
from transformers import AutoTokenizer
from custom_gpt2.gpt2 import GPT, gpt2_generate


def setup_logging(output_dir: str, prefix: str) -> logging.Logger:
    """
    Set up a logger that writes to both the console and a timestamped log file.

    A new log file is created per run using the format:
        <output_dir>/<prefix>_<YYYYMMDD_HHMMSS>.log

    This prevents any previous run's log from being overwritten.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger("ig2")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if setup_logging is called more than once
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- file handler (DEBUG and above → full detail in the log file) ---
    fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # --- console handler (INFO and above → clean terminal output) ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logging initialised. Log file: %s", log_filename)
    return logger


def process_example(example, max_seq_length, tokenizer):
    # example is of the form: [`sentence`, `gold_demo`, `metadata tag`]
    # returns {tokenized_sentence, input_ids, tokenized_completion, output_ids}

    tokenized_sentence = tokenizer.tokenize(example[0])
    input_ids = tokenizer.encode(example[0])

    tokenized_completion = tokenizer.tokenize(example[1])
    output_ids = tokenizer.encode(example[1])

    return {
        "tokenized_sentence": tokenized_sentence,
        "input_ids": input_ids,
        "tokenized_completion": tokenized_completion,
        "output_ids": output_ids,
    }


def scaled_input(emb, batch_size, num_batch):
    baseline = torch.zeros_like(emb)
    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)
    return res, step[0]


def convert_to_triplet_ig2(ig2_list):
    ig2 = np.array(ig2_list)
    max_ig2 = ig2.max()
    ig2_triplet = []
    for i in range(ig2.shape[0]):
        for j in range(ig2.shape[1]):
            if ig2[i][j] >= max_ig2 * 0.1:
                ig2_triplet.append([i, j, float(ig2[i][j])])
    return ig2_triplet


def convert_to_triplet_ig2_gap(ig2_list):
    ig2 = np.array(ig2_list)
    max_ig2 = np.abs(ig2).max()
    ig2_triplet = []
    for i in range(ig2.shape[0]):
        for j in range(ig2.shape[1]):
            if abs(ig2[i][j]) >= max_ig2 * 0.1:
                ig2_triplet.append([i, j, float(ig2[i][j])])
    return ig2_triplet


def safe_open_jsonl(path: str, logger: logging.Logger):
    """
    Open a jsonlines file for writing.  If the file already exists the run is
    aborted early so that previous results are never silently overwritten.
    Raises FileExistsError with a clear message so the caller can handle it.
    """
    if os.path.exists(path):
        msg = (
            f"Output file already exists and would be overwritten: {path}\n"
            "Rename or delete the existing file, or choose a different --output_dir."
        )
        logger.error(msg)
        raise FileExistsError(msg)
    logger.debug("Opening output file for writing: %s", path)
    return jsonlines.open(path, "w")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--demographic_dimension", default=None, type=str, required=True)
    parser.add_argument("--demographic1", default=None, type=str, required=True)
    parser.add_argument("--demographic2", default=None, type=str, required=True)
    parser.add_argument("--modifier", default=None, type=str, required=True)
    parser.add_argument("--gpt2_variant", default="gpt2", type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--pool_activation", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", type=int, default=-1)
    parser.add_argument("--get_pred", action="store_true")
    parser.add_argument("--get_ig2_pred", action="store_true")
    parser.add_argument("--get_ig2_gold", action="store_true")
    parser.add_argument("--get_ig2_gold_filtered", action="store_true")
    parser.add_argument("--get_base", action="store_true")
    parser.add_argument("--get_ig2_gold_gap_filtered", action="store_true")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_batch", default=10, type=int)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_prefix = "Modifier-" + f"{args.demographic_dimension}_mini" + "-" + args.modifier

    # ------------------------------------------------------------------ #
    #  Logging — must happen after output_dir is created                  #
    # ------------------------------------------------------------------ #
    logger = setup_logging(args.output_dir, output_prefix)

    # Log all resolved arguments so the log file is fully self-contained
    logger.info("Run arguments:\n%s", json.dumps(vars(args), indent=2, sort_keys=True))

    # ------------------------------------------------------------------ #
    #  Device selection                                                    #
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = torch.device("cuda:%s" % args.gpus)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger.debug("Random seeds set to %d", args.seed)

    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    demo1_data_path = os.path.join(
        args.data_path, f"{args.demographic_dimension}_mini",
        args.demographic1 + "_" + args.modifier + "_data.json",
    )
    demo2_data_path = os.path.join(
        args.data_path, f"{args.demographic_dimension}_mini",
        args.demographic2 + "_" + args.modifier + "_data.json",
    )
    demo1_tmp_data_path = os.path.join(
        args.data_path, f"{args.demographic_dimension}_mini",
        args.demographic1 + "_" + args.modifier + "_allbags.json",
    )
    demo2_tmp_data_path = os.path.join(
        args.data_path, f"{args.demographic_dimension}_mini",
        args.demographic2 + "_" + args.modifier + "_allbags.json",
    )

    logger.debug("demo1 data path : %s", demo1_data_path)
    logger.debug("demo2 data path : %s", demo2_data_path)
    logger.debug("demo1 allbags cache: %s", demo1_tmp_data_path)
    logger.debug("demo2 allbags cache: %s", demo2_tmp_data_path)

    json.dump(
        args.__dict__,
        open(os.path.join(args.output_dir, output_prefix + ".args.json"), "w"),
        sort_keys=True,
        indent=2,
    )

    # ------------------------------------------------------------------ #
    #  Tokenizer & model                                                   #
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info("Tokenizer loaded (gpt2)")

    logger.info("Clearing GPU cache …")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    model = GPT.from_pretrained(args.gpt2_variant)
    model.to(device)
    model.eval()
    logger.info("Model '%s' loaded and moved to %s", args.gpt2_variant, device)

    # ------------------------------------------------------------------ #
    #  Data loading / caching                                             #
    # ------------------------------------------------------------------ #
    def load_or_build_bags(data_path, tmp_path, label):
        if os.path.exists(tmp_path):
            logger.info("[%s] Loading allbags cache from: %s", label, tmp_path)
            with open(tmp_path, "r") as f:
                return json.load(f)

        logger.info("[%s] Cache not found. Building allbags from: %s", label, data_path)
        with open(data_path, "r") as f:
            eval_bag_list_all = json.load(f)

        bags_perrel = {}
        for eval_bag in eval_bag_list_all:
            bag_rel = eval_bag[0][2].split("(")[0]
            if bag_rel not in bags_perrel:
                bags_perrel[bag_rel] = []
            if args.debug != -1 and len(bags_perrel[bag_rel]) >= args.debug:
                continue
            bags_perrel[bag_rel].append(eval_bag)

        with open(tmp_path, "w") as fw:
            json.dump(bags_perrel, fw, indent=2)
        logger.info("[%s] Allbags cache written to: %s", label, tmp_path)
        return bags_perrel

    demo1_eval_bag_list_perrel = load_or_build_bags(
        demo1_data_path, demo1_tmp_data_path, args.demographic1
    )
    demo2_eval_bag_list_perrel = load_or_build_bags(
        demo2_data_path, demo2_tmp_data_path, args.demographic2
    )
    logger.info("Data read and cached in allbags files.")

    # ------------------------------------------------------------------ #
    #  IG2 calculation                                                     #
    # ------------------------------------------------------------------ #
    for demo_label, eval_bag_list_perrel in [
        (args.demographic1, demo1_eval_bag_list_perrel),
        (args.demographic2, demo2_eval_bag_list_perrel),
    ]:
        for relation, eval_bag_list in eval_bag_list_perrel.items():
            tic = time.perf_counter()

            output_file = os.path.join(
                args.output_dir, output_prefix + "-" + demo_label + ".rlt.jsonl"
            )

            # Guard against silent overwrites
            with safe_open_jsonl(output_file, logger) as fw:
                logger.info(
                    "[%s | %s] Processing %d bags → %s",
                    demo_label, relation, len(eval_bag_list), output_file,
                )

                for bag_idx, eval_bag in enumerate(eval_bag_list):
                    logger.info(
                        "[%s | %s] Bag %d/%d", demo_label, relation,
                        bag_idx + 1, len(eval_bag_list),
                    )
                    res_dict_bag = []

                    for eval_example in eval_bag:
                        processed_example = process_example(
                            eval_example, args.max_seq_length, tokenizer
                        )

                        tokenized_sentence = processed_example["tokenized_sentence"]
                        input_ids = processed_example["input_ids"]
                        tokenized_completion = processed_example["tokenized_completion"]
                        output_ids = processed_example["output_ids"]

                        input_ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                        output_ids_t = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)

                        T_p = input_ids_t.size(1)
                        T_c = output_ids_t.size(1)

                        res_dict = {"pred": [], "ig2_pred": [], "ig2_gold": [], "base": []}

                        tokens_info = {
                            "tokenized_sentence": tokenized_sentence,
                            "tokenized_completion": tokenized_completion,
                            "gold_demo": eval_example[1],
                            "metadata": eval_example[2],
                        }

                        gold_full_sequence = torch.cat([input_ids_t, output_ids_t], dim=1)
                        target_positions = list(range(T_p - 1, T_p + T_c - 1))

                        # Initialise here so the layer loop can always reference
                        # these safely regardless of which flags are active.
                        pred_full_sequence = None
                        generated_tokens = None
                        pred_label = None

                        if args.get_pred:
                            pred_log_prob, generated_tokens, pred_full_sequence = gpt2_generate(
                                model, input_ids_t, gen_len=T_c, max_seq_length=args.max_seq_length
                            )
                            res_dict["pred"].append({
                                "pred_log_prob": pred_log_prob.item(),
                                "generated_tokens": generated_tokens.tolist(),
                                "generated_sentence": tokenizer.decode(generated_tokens[0, :].tolist()),
                                "pred_complete_sentence": tokenizer.decode(pred_full_sequence[0, :].tolist()),
                            })
                            logger.debug(
                                "Prediction: '%s'",
                                tokenizer.decode(generated_tokens[0, :].tolist()),
                            )

                        if args.get_ig2_pred:
                            # Only generate again when --get_pred was not set
                            # (pred_full_sequence already holds the result otherwise).
                            if not args.get_pred:
                                with torch.no_grad():
                                    _, generated_tokens, pred_full_sequence = gpt2_generate(
                                        model, input_ids_t, gen_len=T_c,
                                        max_seq_length=args.max_seq_length,
                                    )
                            pred_label = generated_tokens.squeeze(0).tolist()

                        for tgt_layer in range(model.config.n_layer):

                            if args.get_ig2_gold:
                                with torch.no_grad():
                                    _, _, mlp_neurons_gold = model(
                                        gold_full_sequence, target_layer=tgt_layer, return_neurons=True
                                    )

                                gold_completion_neurons = mlp_neurons_gold[:, target_positions, :]

                                if args.pool_activation == "mean":
                                    pooled_neurons_gold = gold_completion_neurons.mean(dim=1)
                                elif args.pool_activation == "max":
                                    raise NotImplementedError
                                else:
                                    raise ValueError("pooling only mean and max.")

                                scaled_neurons_gold, neuron_step_gold = scaled_input(
                                    pooled_neurons_gold, args.batch_size, args.num_batch
                                )
                                scaled_neurons_gold.requires_grad_(True)

                                ig2_gold = None
                                for batch_idx in range(args.num_batch):
                                    batch_neurons = scaled_neurons_gold[
                                        batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size
                                    ]
                                    _, grad = model(
                                        gold_full_sequence,
                                        target_layer=tgt_layer,
                                        patched_mlp_activation=batch_neurons,
                                        target_positions=target_positions,
                                        target_label=output_ids,
                                    )
                                    grad = grad.sum(dim=0)
                                    ig2_gold = grad if ig2_gold is None else torch.add(ig2_gold, grad)

                                ig2_gold = ig2_gold * neuron_step_gold
                                res_dict["ig2_gold"].append(ig2_gold.tolist())
                                logger.debug("Layer %d ig2_gold computed.", tgt_layer)

                            if args.get_ig2_pred and pred_full_sequence is not None:
                                with torch.no_grad():
                                    _, _, mlp_neurons_pred = model(
                                        pred_full_sequence, target_layer=tgt_layer, return_neurons=True
                                    )

                                pred_completion_neurons = mlp_neurons_pred[:, target_positions, :]

                                if args.pool_activation == "mean":
                                    pooled_neurons_pred = pred_completion_neurons.mean(dim=1)
                                elif args.pool_activation == "max":
                                    raise NotImplementedError
                                else:
                                    raise ValueError("pooling only mean and max.")

                                scaled_neurons_pred, neurons_step_pred = scaled_input(
                                    pooled_neurons_pred, args.batch_size, args.num_batch
                                )
                                scaled_neurons_pred.requires_grad_(True)

                                ig2_pred = None
                                for batch_idx in range(args.num_batch):
                                    batch_neurons = scaled_neurons_pred[
                                        batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size
                                    ]
                                    _, grad = model(
                                        pred_full_sequence,
                                        target_layer=tgt_layer,
                                        patched_mlp_activation=batch_neurons,
                                        target_positions=target_positions,
                                        target_label=pred_label,
                                    )
                                    grad = grad.sum(dim=0)
                                    ig2_pred = grad if ig2_pred is None else torch.add(ig2_pred, grad)

                                ig2_pred = ig2_pred * neurons_step_pred
                                res_dict["ig2_pred"].append(ig2_pred.tolist())
                                logger.debug("Layer %d ig2_pred computed.", tgt_layer)

                            if args.get_base:
                                if not args.get_ig2_gold:
                                    with torch.no_grad():
                                        _, _, mlp_neurons_gold = model(
                                            gold_full_sequence, target_layer=tgt_layer, return_neurons=True
                                        )
                                    gold_completion_neurons = mlp_neurons_gold[:, target_positions, :]
                                    pooled_neurons_gold = gold_completion_neurons.mean(dim=1)

                                res_dict["base"].append(pooled_neurons_gold.squeeze().tolist())
                                logger.debug("Layer %d base activations collected.", tgt_layer)

                        if args.get_ig2_gold_filtered:
                            res_dict["ig2_gold_filtered"] = convert_to_triplet_ig2(res_dict["ig2_gold"])
                            logger.debug(
                                "ig2_gold_filtered: %d triplets retained.",
                                len(res_dict["ig2_gold_filtered"]),
                            )

                        res_dict_bag.append([tokens_info, res_dict])

                    fw.write(res_dict_bag)

            toc = time.perf_counter()
            logger.info(
                "Demo: %s | Relation: %s — done in %.4f s",
                demo_label, relation, toc - tic,
            )

    # ------------------------------------------------------------------ #
    #  Gap computation                                                     #
    # ------------------------------------------------------------------ #
    for demo1_relation, demo2_relation in zip(
        demo1_eval_bag_list_perrel.keys(), demo2_eval_bag_list_perrel.keys()
    ):
        gap_output_file = os.path.join(
            args.output_dir,
            output_prefix
            + "-filtered-gap-rm-base-"
            + args.demographic1
            + "-"
            + args.demographic2
            + ".rlt.jsonl",
        )

        with (
            jsonlines.open(
                os.path.join(args.output_dir, output_prefix + "-" + args.demographic1 + ".rlt.jsonl"), "r"
            ) as fb,
            jsonlines.open(
                os.path.join(args.output_dir, output_prefix + "-" + args.demographic2 + ".rlt.jsonl"), "r"
            ) as fw,
            safe_open_jsonl(gap_output_file, logger) as filf_rmb_gap,
        ):
            tic = time.perf_counter()
            logger.info(
                "Computing gap for relation pair: %s — %s",
                demo1_relation, demo2_relation,
            )

            for demo1_res_dict_bag, demo2_res_dict_bag in zip(fb, fw):
                gap_res_dict_rmb_bag = []
                for demo1_example, demo2_example in zip(demo1_res_dict_bag, demo2_res_dict_bag):
                    demo1_res_dict, demo2_res_dict = demo1_example[1], demo2_example[1]
                    demo1_tokens_info = demo1_example[0]

                    demo1_ig2_gold = np.array(demo1_res_dict["ig2_gold"], np.float32)
                    demo2_ig2_gold = np.array(demo2_res_dict["ig2_gold"], np.float32)

                    ig2_gold_gap = (demo1_ig2_gold - demo2_ig2_gold).tolist()

                    gap_tokens_info = {
                        "tokenized_sentence": demo1_tokens_info["tokenized_sentence"],
                        "gap_relation": demo1_relation + "-" + demo2_relation,
                        "gold_obj": args.demographic1 + " - " + args.demographic2,
                    }
                    gap_res_rmb_dict = {"ig2_gold_gap": ig2_gold_gap}

                    if args.get_ig2_gold_gap_filtered:
                        gap_res_rmb_dict["ig2_gold_gap"] = convert_to_triplet_ig2_gap(
                            gap_res_rmb_dict["ig2_gold_gap"]
                        )
                        logger.debug(
                            "ig2_gold_gap_filtered: %d triplets retained.",
                            len(gap_res_rmb_dict["ig2_gold_gap"]),
                        )

                    gap_res_dict_rmb_bag.append([gap_tokens_info, gap_res_rmb_dict])

                filf_rmb_gap.write(gap_res_dict_rmb_bag)

            toc = time.perf_counter()
            logger.info(
                "Gap relation %s — done in %.4f s",
                demo1_relation + "-" + demo2_relation, toc - tic,
            )

    logger.info("All done.")


if __name__ == "__main__":
    main()