# TODO: to add multi-gpu support
# TODO: model the probability of the sentence and not the final token only (provide the abiluty to run both version)

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import json
import jsonlines
from transformers import AutoTokenizer
from custom_gpt2.gpt2 import GPT, gpt2_generate

def process_example(example, max_seq_length, tokenizer):
    # example is of the form: [`sentence`, `gold_demo`, `metadata tag`]
    # returns {tokenized_sentence, input_ids, tokenized_completion, output_ids}
    
    # sentence tokenization (X)
    tokenized_sentence = tokenizer.tokenize(example[0])
    input_ids = tokenizer.encode(example[0])
    
    # completions
    tokenized_completion = tokenizer.tokenize(example[1])
    output_ids = tokenizer.encode(example[1])
    
    return {
        "tokenized_sentence": tokenized_sentence,
        "input_ids": input_ids,
        "tokenized_completion": tokenized_completion,
        "output_ids": output_ids
    }
    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path.")
    parser.add_argument("--demographic_dimension",
                        default=None,
                        type=str,
                        required=True,
                        help="The demographic dimension, such as Ethnicity, Gender, etc.")
    parser.add_argument("--demographic1",
                        default=None,
                        type=str,
                        required=True,
                        help="The first demographic for the gap computation, such as black.")
    parser.add_argument("--demographic2",
                        default=None,
                        type=str,
                        required=True,
                        help="The second demographic for the gap computation, such as white.")
    parser.add_argument("--modifier",
                        default=None,
                        type=str,
                        required=True,
                        help="The modifier type for the gap computation, such as negative (N).")
    parser.add_argument("--gpt2_variant",
                        default="gpt2",
                        type=str,
                        help="gpt2 variant to do analysis on (gpt2, gpt2-medium, gpt2-large, gpt2-xl)")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=1024,
                        type=int,
                        help="Default is 1024 which is the context window for the GPT2 model.")
    # parser.add_argument("--no_cuda",
    #                     default=False,
    #                     action='store_true',
    #                     help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--debug",
                        type=int,
                        default=-1,
                        help="How many examples to debug. -1 denotes no debugging")
    parser.add_argument("--get_pred",
                        action='store_true',
                        help="Whether to get prediction results.")
    parser.add_argument("--get_ig2_pred",
                        action='store_true',
                        help="Whether to get integrated gradient at the predicted label.")
    parser.add_argument("--get_ig2_gold",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label.")
    parser.add_argument("--get_ig2_gold_filtered",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label after filtering.")
    parser.add_argument("--get_base",
                        action='store_true',
                        help="Whether to get base values. ")
    # parser.add_argument("--get_base_filtered",
    #                     action='store_true',
    #                     help="Whether to get base values after filtering. ")
    parser.add_argument("--get_ig2_gold_gap_filtered",
                        action='store_true',
                        help="Whether to get integrated gradient gap at the gold label after filtering.")
    # parser.add_argument("--get_base_gap_filtered",
    #                     action='store_true',
    #                     help="Whether to get base values gap after filtering. ")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")

    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda:%s" % args.gpus)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"device: {device}")
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    demo1_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                   args.demographic1 + '_' + args.modifier + '_data.json')
    demo2_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                   args.demographic2 + '_' + args.modifier + '_data.json')

    demo1_tmp_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                       args.demographic1 + '_' + args.modifier + '_allbags.json')
    demo2_tmp_data_path = os.path.join(args.data_path, args.demographic_dimension,
                                       args.demographic2 + '_' + args.modifier + '_allbags.json')

    output_prefix = 'Modifier-' + args.demographic_dimension + '-' + args.modifier

    json.dump(args.__dict__, open(os.path.join(args.output_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("*** EMPTY GPU CACHE ***")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    model = GPT.from_pretrained(args.gpt2_variant)
    model.to(device)
    
    model.eval()
        
    if os.path.exists(demo1_tmp_data_path):
        with open(demo1_tmp_data_path, 'r') as f:
            demo1_eval_bag_list_perrel = json.load(f)
    else:
        with open(demo1_data_path, 'r') as f:
            demo1_eval_bag_list_all = json.load(f)
        demo1_eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(demo1_eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in demo1_eval_bag_list_perrel:
                demo1_eval_bag_list_perrel[bag_rel] = []
            if args.debug != -1 and len(demo1_eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            demo1_eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(demo1_tmp_data_path, 'w') as fw:
            json.dump(demo1_eval_bag_list_perrel, fw, indent=2)

    # demo2
    if os.path.exists(demo2_tmp_data_path):
        with open(demo2_tmp_data_path, 'r') as f:
            demo2_eval_bag_list_perrel = json.load(f)
    else:
        print(f"{demo2_data_path}")
        with open(demo2_data_path, 'r') as f:
            demo2_eval_bag_list_all = json.load(f)
        demo2_eval_bag_list_perrel = {}
        for bag_idx, eval_bag in enumerate(demo2_eval_bag_list_all):
            bag_rel = eval_bag[0][2].split('(')[0]
            if bag_rel not in demo2_eval_bag_list_perrel:
                demo2_eval_bag_list_perrel[bag_rel] = []
            if args.debug != -1 and len(demo2_eval_bag_list_perrel[bag_rel]) >= args.debug:
                continue
            demo2_eval_bag_list_perrel[bag_rel].append(eval_bag)
        with open(demo2_tmp_data_path, 'w') as fw:
            json.dump(demo2_eval_bag_list_perrel, fw, indent=2)
            
    print(f"Data read and cached in allbags files.")
    
    
    # IG2 calculation
    for relation, eval_bag_list in demo1_eval_bag_list_perrel.items():
        tic = time.perf_counter()
        with jsonlines.open(os.path.join(args.output_dir, output_prefix + '-' + args.demographic1 + '.rlt' + '.jsonl'), 'w') as demo1_fw:
            for bag_idx, eval_bag in enumerate(eval_bag_list):
                res_dict_bag = []
                for eval_example in eval_bag:
                    # eval_example: [`sentence`, `gold_demography`, `metadata-tag`]
                    processed_example = process_example(eval_example, args.max_seq_length, tokenizer)
                    
                    tokenized_example, input_ids, tokenized_completion, output_ids = processed_example["tokenized_example"], processed_example["input_ids"], processed_example["tokenized_completion"], processed_example["output_ids"]
                    
                    # unsqueeze to create batch dimension
                    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                    output_ids = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0)
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)

                    # autoregressive model and hence we are looking at the end
                    # which means the tgt_pos is always the end
                    # NOTE: this is still v1 and hence we are modelling the probability of the final token and not the sentence

                    res_dict = {
                        'pred': [],
                        'ig2_pred': [],
                        'ig2_gold': [],
                        'base': []
                    }

                    if args.get_pred:
                        # gets the prediction probability of the next token
                        # NOTE: generating output_ids.size(1)-many tokens for fair comparison
                        pred_log_prob, generated_tokens, full_sequence = gpt2_generate(model, input_ids, gen_len=output_ids.size(1), max_seq_length=args.max_seq_length)
                        
                        res_dict['pred'].append({
                                "pred_log_prob": pred_log_prob,
                                "generated_tokens": generated_tokens.tolist(),
                                "generated_sentence": tokenizer.decode(generated_tokens[0, :].tolist()),
                                "complete_sentence": tokenizer.decode(full_sequence[0, :].tolist())
                            }
                        )

                    # TODO: pick up work from here
                    for tgt_layer in range(model.bert.config.num_hidden_layers):
                        ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)  # (1, ffn_size), (1, n_vocab)
                        pred_label = int(torch.argmax(logits[0, :]))  # scalar, 这里获得了pred_label，方便后面计算ig2_pred
                        gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])
                        tokens_info['pred_obj'] = tokenizer.convert_ids_to_tokens(pred_label)
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, ffn_size), (ffn_size)
                        scaled_weights.requires_grad_(True)

                        if args.get_ig2_pred:
                            ig2_pred = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)  # (batch, n_vocab), (batch, ffn_size)
                                grad = grad.sum(dim=0)  # (ffn_size)
                                ig2_pred = grad if ig2_pred is None else torch.add(ig2_pred, grad)  # (ffn_size)
                            ig2_pred = ig2_pred * weights_step  # (ffn_size)
                            res_dict['ig2_pred'].append(ig2_pred.tolist())

                        if args.get_ig2_gold:
                            ig2_gold = None
                            for batch_idx in range(args.num_batch):
                                batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                                _, grad = model(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer,
                                                tmp_score=batch_weights, tgt_label=gold_label)
                                grad = grad.sum(dim=0)
                                ig2_gold = grad if ig2_gold is None else torch.add(ig2_gold, grad)
                            ig2_gold = ig2_gold * weights_step
                            res_dict['ig2_gold'].append(ig2_gold.tolist())

                        if args.get_base:
                            res_dict['base'].append(ffn_weights.squeeze().tolist())

                    if args.get_ig2_gold_filtered:
                        res_dict['ig2_gold'] = convert_to_triplet_ig2(res_dict['ig2_gold'])

                    res_dict_bag.append([tokens_info, res_dict])

                demo1_fw.write(res_dict_bag)

        toc = time.perf_counter()
        print(f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****")
            
if __name__ == "__main__":
    main()