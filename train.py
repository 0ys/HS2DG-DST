from __future__ import absolute_import, division, print_function

from setproctitle import setproctitle
setproctitle('0ys_HS2DG-DST')

import argparse
import json
import logging
import math
import os
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)

from tqdm import tqdm
from models.generator import Generator
from utils import helper
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.constant import track_slots, ansvocab, slot_map, n_slot, TURN_SPLIT, TEST_TURN_SPLIT, ontology_value_list
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label
from evaluation import op_evaluation, model_evaluation, joint_evaluation
# from transformers.configuration_albert import AlbertConfig
from transformers import AlbertConfig
# from transformers.tokenization_albert import AlbertTokenizer
from transformers import AlbertTokenizer
from transformers.optimization import AdamW
# from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import get_linear_schedule_with_warmup
from utils.logger import get_logger

import sys
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

csv.field_size_limit(sys.maxsize)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

'''compute joint operation scores based on logits of two stages 
'''
def compute_jointscore(start_scores, end_scores, gen_scores, pred_ops, ans_vocab, slot_mask):
    seq_lens = start_scores.shape[-1]
    joint_score = start_scores.unsqueeze(-2) + end_scores.unsqueeze(-1)
    triu_mask = np.triu(np.ones((joint_score.size(-1), joint_score.size(-1))))
    triu_mask[0, 1:] = 0
    triu_mask = (torch.Tensor(triu_mask) ==  0).bool()
    joint_score = joint_score.masked_fill(triu_mask.unsqueeze(0).unsqueeze(0).cuda(),-1e9).masked_fill(
        slot_mask.unsqueeze(1).unsqueeze(-2)  == 0, -1e9)
    joint_score = F.softmax(joint_score.view(joint_score.size(0), joint_score.size(1), -1),
                            dim = -1).view(joint_score.size(0), joint_score.size(1), seq_lens, -1)

    score_diff = (joint_score[:, :, 0, 0] - joint_score[:, :, 1:, 1:].max(dim = -1)[0].max(dim = -1)[
        0])
    score_noans = pred_ops[:, :, -1] - pred_ops[:, :, 0]
    slot_ans_count = (ans_vocab.sum(-1) != 0).sum(dim=-1)-2
    ans_idx = torch.where(slot_ans_count < 0, torch.zeros_like(slot_ans_count), slot_ans_count)
    neg_ans_mask = torch.cat((torch.linspace(0, ans_vocab.size(0) - 1,
                                             ans_vocab.size(0)).unsqueeze(0).long(),
                              ans_idx.unsqueeze(0)),
                             dim = 0)
    neg_ans_mask = torch.sparse_coo_tensor(neg_ans_mask, torch.ones(ans_vocab.size(0)),
                                           (ans_vocab.size(0),
                                            ans_vocab.size(1))).to_dense().cuda()
    score_neg = gen_scores.masked_fill(neg_ans_mask.unsqueeze(0) == 0, -1e9).max(dim=-1)[0]
    score_has = gen_scores.masked_fill(neg_ans_mask.unsqueeze(0) == 1, -1e9).max(dim=-1)[0]
    cate_score_diff = score_neg - score_has
    score_diffs = score_diff.view(-1).cpu().detach().numpy().tolist()
    cate_score_diffs = cate_score_diff.view(-1).cpu().detach().numpy().tolist()
    score_noanses = score_noans.view(-1).cpu().detach().numpy().tolist()
    return score_diffs, cate_score_diffs, score_noanses


def saveOperationLogits(model, device, dataset, save_path, turn):
    score_ext_map = {}
    model.eval()
    for batch in tqdm(dataset, desc = "Evaluating"):
        batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in batch]
        input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, sid = batch
        batch_size = input_ids.shape[0]
        seq_lens = input_ids.shape[1]
        start_logits, end_logits, has_ans, gen_scores, _, _, _ = model(input_ids = input_ids,
                                                                       token_type_ids = segment_ids,
                                                                       state_positions = state_position_ids,
                                                                       attention_mask = input_mask,
                                                                       slot_mask = slot_mask)

        score_ext = has_ans.cpu().detach().numpy().tolist()
        for i, sd in enumerate(score_ext):
            score_ext_map[sid[i]] = sd
    with open(os.path.join(save_path, "cls_score_test_turn{}.json".format(turn)), "w") as writer:
        writer.write(json.dumps(score_ext_map, indent = 4) + "\n")

def compute_span_loss(gen_ids, input_ids, start_scores, end_scores, sample_mask):
    # print(f'gen_ids:{gen_ids}')
    # print(f'input_ids:{input_ids}') #torch.Size([1, 512])
    # print(f'start_scores:{start_scores.shape}') #torch.Size([1, 30, 511])
    # print(f'end_scores:{end_scores.shape}') #torch.Size([1, 30, 511])
    # print(f'sample_mask:{sample_mask}')
    loss = 0
    for i in range(start_scores.shape[0]):
        batch_mask = sample_mask[i]
        # print(f'batch_mask:{batch_mask}')
        start_idx = [-1 for i in range(n_slot)]
        end_idx = [-1 for i in range(n_slot)]
        for ti in range(n_slot):
            if not batch_mask[ti]:
                continue
            value = gen_ids[i][ti][0]
            value = value[:-1] if isinstance(value, list) else [value]
            # print(f'value:{value}')
            batch_input = input_ids[i].cpu().detach().numpy().tolist()
            for text_idx in range(len(input_ids[i]) - len(value)):
                if batch_input[text_idx: text_idx + len(value)] == value:
                    start_idx[ti] = text_idx
                    end_idx[ti] = text_idx + len(value) - 1
                    break
        start_idx = torch.from_numpy(np.array(start_idx)).cuda()
        end_idx = torch.from_numpy(np.array(end_idx)).cuda()
        # print(f'start_idx:{start_idx}')
        # print(f'end_idx:{end_idx}')
        loss += masked_cross_entropy_for_value(start_scores[i].contiguous(),
                                                start_idx.contiguous(),
                                                sample_mask = batch_mask,
                                                pad_idx = -1
                                                )
        batch_loss = masked_cross_entropy_for_value(end_scores[i].contiguous(),
                                                end_idx.contiguous(),
                                                sample_mask=batch_mask,
                                                pad_idx=-1
                                                )
        loss += batch_loss
    loss /= start_scores.shape[0]
    return loss

def masked_cross_entropy_for_value(logits, target, sample_mask = None, slot_mask = None, pad_idx = -1):
    mask = logits.eq(0)
    pad_mask = target.ne(pad_idx)
    target = target.masked_fill(target < 0, 0)
    sample_mask = pad_mask & sample_mask if sample_mask is not None else pad_mask
    sample_mask = slot_mask & sample_mask if slot_mask is not None else sample_mask
    target = target.masked_fill(sample_mask ==  0, 0)
    logits = logits.masked_fill(mask, 1)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim = 1, index = target_flat)
    losses = losses_flat.view(*target.size())
    # if mask is not None:
    sample_num = sample_mask.sum().float()
    losses = losses * sample_mask.float()
    loss = (losses.sum() / sample_num) if sample_num != 0 else losses.sum()
    return loss

def addSpecialTokens(tokenizer, specialtokens):
    special_key = "additional_special_tokens"
    tokenizer.add_special_tokens({special_key: specialtokens})

def fixontology(ontology, turn, tokenizer):
    ans_vocab = []
    esm_ans_vocab = []
    esm_ans = ansvocab
    slot_mm = np.zeros((len(slot_map), len(esm_ans)))
    max_anses_length = 0
    max_anses = 0
    for i, k in enumerate(ontology.keys()):
        if k in track_slots:
            s = ontology[k]
            s['name'] = k
            if not s['type']:
                s['db'] = []
            slot_mm[i][slot_map[s['name']]] = 1
            ans_vocab.append(s)
    for si in esm_ans:
        slot_anses = []
        for ans in si:
            enc_ans=tokenizer.encode(ans)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(enc_ans)
        max_anses = max(max_anses, len(slot_anses))
        esm_ans_vocab.append(slot_anses)
    for s in esm_ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    esm_ans_vocab = np.array(esm_ans_vocab)
    ans_vocab_tensor = torch.from_numpy(esm_ans_vocab)
    slot_mm = torch.from_numpy(slot_mm).float()
    return ans_vocab, slot_mm, ans_vocab_tensor

def mask_ans_vocab(ontology, slot_meta, tokenizer):
    ans_vocab = []
    max_anses = 0
    max_anses_length = 0
    change_k = []
    cate_mask = []
    for k in ontology:
        if (' range' in k['name']) or (' at' in k['name']) or (' by' in k['name']):
            change_k.append(k)
    for key in change_k:
        new_k = key['name'].replace(' ', '')
        key['name'] = new_k
    for s in ontology:
        cate_mask.append(s['type'])
        v_list = s['db']
        slot_anses = []
        for v in v_list:
            ans = tokenizer.encode(v)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(ans)
        max_anses = max(max_anses, len(slot_anses))
        ans_vocab.append(slot_anses)
    for s in ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    ans_vocab = np.array(ans_vocab)
    ans_vocab_tensor = torch.from_numpy(ans_vocab)
    return ans_vocab_tensor, ans_vocab, cate_mask

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type", default = 'albert', type = str,
                        help = "Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default = 'pretrained_models/albert-base/', type = str,
                        help = "Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default = "saved_models/", type = str,
                        help = "The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--config_name", default = "", type = str,
                        help = "Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default = "", type = str,
                        help = "Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default = "", type = str,
                        help = "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", default = False, action = 'store_true',
                        help = "Whether to run training.")
    parser.add_argument("--evaluate_during_training", action = 'store_true',
                        help = "Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action = 'store_true',
                        help = "Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default = 32, type = int,
                        help = "Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default = 32, type = int,
                        help = "Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default = 3e-5, type = float,
                        help = "The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default = 0.1, type = float,
                        help = "Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default = 1e-8, type = float,
                        help = "Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help = "Max gradient norm.")
    parser.add_argument("--num_train_epochs", default = 3.0, type = float,
                        help = "Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default = -1, type = int,
                        help = "If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default = 0, type = int,
                        help = "Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type = int, default = 50,
                        help = "Log every X updates steps.")
    parser.add_argument('--save_steps', type = int, default = 50,
                        help = "Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action = 'store_true',
                        help = "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action = 'store_true',
                        help = "Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default = True, action = 'store_true',
                        help = "Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action = 'store_true',
                        help = "Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type = int, default = 42,
                        help = "random seed for initialization")

    parser.add_argument('--fp16', action = 'store_true',
                        help = "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type = str, default = 'O1',
                        help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--local_rank", type = int, default = -1,
                        help = "For distributed training: local_rank")  # DST params
    parser.add_argument("--data_root", default = 'data/mwz2.2/', type = str)
    parser.add_argument("--train_data", default = 'train_dials.json', type = str)
    parser.add_argument("--dev_data", default = 'dev_dials.json', type = str)
    parser.add_argument("--test_data", default = 'test_dials.json', type = str)
    parser.add_argument("--ontology_data", default = 'schema.json', type = str)
    parser.add_argument("--vocab_path", default = 'assets/vocab.txt', type = str)
    parser.add_argument("--save_dir", default = 'saved_models', type = str)
    parser.add_argument("--load_model", default = False, action = 'store_true')
    parser.add_argument("--load_best", default = False, action = 'store_true')
    parser.add_argument("--load_ckpt_epoch", default='checkpoint_epoch_9.bin', type=str)
    parser.add_argument("--load_test_op_data_path", default='cls_score_test_state_update_predictor_output.json', type=str)
    parser.add_argument("--random_seed", default = 42, type = int)
    parser.add_argument("--num_workers", default = 4, type = int)
    parser.add_argument("--batch_size", default = 1, type = int)
    parser.add_argument("--enc_warmup", default = 0.01, type = float)
    parser.add_argument("--dec_warmup", default = 0.01, type = float)
    parser.add_argument("--enc_lr", default = 5e-6, type = float)
    parser.add_argument("--base_lr", default = 1e-4, type = float)
    parser.add_argument("--n_epochs", default = 10, type = int)
    parser.add_argument("--eval_epoch", default = 1, type = int)
    parser.add_argument("--eval_step", default=5, type=int)
    parser.add_argument("--turn", default = 2, type = int)
    parser.add_argument("--op_code", default = "2", type = str)
    parser.add_argument("--slot_token", default = "[SLOT]", type = str)
    parser.add_argument("--dropout", default = 0.0, type = float)
    parser.add_argument("--hidden_dropout_prob", default = 0.0, type = float)
    parser.add_argument("--attention_probs_dropout_prob", default = 0.1, type = float)
    parser.add_argument("--decoder_teacher_forcing", default = 0.5, type = float)
    parser.add_argument("--word_dropout", default = 0.1, type = float)
    parser.add_argument("--not_shuffle_state", default = True, action = 'store_true')

    parser.add_argument("--n_history", default = 1, type = int)
    parser.add_argument("--max_seq_length", default = 256, type = int)
    parser.add_argument("--sketch_weight", default = 0.55, type = float)
    parser.add_argument("--answer_weight", default = 0.6, type = float)
    parser.add_argument("--generation_weight", default = 0.2, type = float)
    parser.add_argument("--extraction_weight", default = 0.1, type = float)
    parser.add_argument("--msg", default = None, type = str)

    # "graph_model"
    parser.add_argument("--graph_type", default = "GAT", type = str)
    parser.add_argument("--num_layer", default = 4, type = int)
    parser.add_argument("--num_head", default = 4, type = int)
    parser.add_argument("--feature_size", default = 768, type = int)
    parser.add_argument("--num_hop", default = 2, type = int)
    parser.add_argument("--graph_dropout", default = 0.2, type = float)
    parser.add_argument("--graph_mode", default = "full", type = str)
    parser.add_argument("--graph_residual", default = False, action = 'store_true')
    parser.add_argument("--cls_loss", default = False, action = 'store_true')
    parser.add_argument("--connect_type", default = 'ds_value_only', type = str)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl')
        args.n_gpu = 1
    args.device = device

    print(f'ProcessRank:{args.local_rank}, device:{device}, n_gpu:{args.n_gpu}, distributed training:{bool(args.local_rank != -1)}, 16-bits training:{args.fp16},')

    # Set seed
    set_seed(args)
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    turn = args.turn
    
    print(f'dataset: {args.data_root}')
    print(f'save_dir: {args.save_dir}')
    ontology = json.load(open(args.data_root + args.ontology_data))

    slot_meta = make_slot_meta(ontology)
    
    with torch.cuda.device(0):
        op2id = OP_SET[args.op_code]
        print(op2id)
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path + "spiece.model")
        addSpecialTokens(tokenizer, ['[SLOT]', '[NULL]', '[EOS]', '[dontcare]'])
        args.vocab_size = len(tokenizer)
        ontology, slot_mm, esm_ans_vocab = fixontology(slot_meta, turn, tokenizer)
        _, ans_vocab_nd, cate_mask = mask_ans_vocab(ontology, slot_meta, tokenizer)
        
        # test_op_data_path = args.data_root + "cls_score_test_state_update_predictor_output.json"
        test_op_data_path = args.data_root + args.load_test_op_data_path
        isfilter = True

        model = Generator(args, esm_ans_vocab, slot_mm, turn=turn)
        train_data_raw, _, _ = prepare_dataset(data_path = args.data_root + args.train_data,
                                               tokenizer = tokenizer,
                                               slot_meta = slot_meta,
                                               n_history = args.n_history,
                                               max_seq_length = args.max_seq_length,
                                               op_code = args.op_code,
                                               slot_ans = ontology,
                                               turn = turn,
                                               op_data_path = None,
                                               isfilter = isfilter
                                               )
        train_data = MultiWozDataset(train_data_raw,
                                     tokenizer,
                                     slot_meta,
                                     args.max_seq_length,
                                     ontology,
                                     args.word_dropout,
                                     turn = turn)
        print("# train examples %d" % len(train_data_raw))

        dev_data_raw, _, _ = prepare_dataset(data_path = args.data_root + args.dev_data,
                                                 tokenizer = tokenizer,
                                                 slot_meta = slot_meta,
                                                 n_history = args.n_history,
                                                 max_seq_length = args.max_seq_length,
                                                 op_code = args.op_code,
                                                 turn = turn,
                                                 slot_ans = ontology,
                                                 op_data_path = None,
                                                 isfilter = False)
        dev_data = MultiWozDataset(dev_data_raw,
                                   tokenizer,
                                   slot_meta,
                                   args.max_seq_length,
                                   ontology,
                                   word_dropout = 0,
                                   turn = turn)
        print("# dev examples %d" % len(dev_data_raw))
        # print(dev_data_raw[0])

        test_data_raw, _, _ = prepare_dataset(data_path = args.data_root + args.test_data,
                                            tokenizer = tokenizer,
                                            slot_meta = slot_meta,
                                            n_history = args.n_history,
                                            max_seq_length = args.max_seq_length,
                                            op_code = args.op_code,
                                            turn = turn,
                                            slot_ans = ontology,
                                            op_data_path = test_op_data_path,
                                            isfilter = False)
        print("# test examples %d" % len(test_data_raw))
        # print(test_data_raw[0])

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler = train_sampler,
                                      batch_size = args.batch_size,
                                      collate_fn = train_data.collate_fn,
                                      num_workers = args.num_workers,
                                      worker_init_fn = worker_init_fn)

        dev_sampler = RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    sampler=dev_sampler,
                                    batch_size=args.batch_size,
                                    collate_fn=dev_data.collate_fn,
                                    num_workers=args.num_workers,
                                    worker_init_fn=worker_init_fn)

        if args.load_model:
            if args.load_best:
                checkpoint = torch.load(os.path.join(args.save_dir, args.load_ckpt_epoch))
                model.load_state_dict(checkpoint['model'])
            else:
                checkpoint = torch.load(os.path.join(args.save_dir, args.load_ckpt_epoch))
                model.load_state_dict(checkpoint)

        model.to(args.device)
        # print("Training/evaluation parameters %s", args)

        # - Value Graph
        if args.graph_mode != 'none':
            print("make Value Graph Ontology..")
            value_id2tokenized_text = {}
            ontology_value_id2tokenized_text = {}
            ds_list = []
            value_id2text = {}
            ontology_value_text2id = {}
            ontology_value_id2text = {}
            for ds in ontology:
                ds_list.append(ds['name'])
                value_id2text[ds['name']] = {}
                value_id2tokenized_text[ds['name']] = {}
                value_dict = ds['db']
                # print(ds['name'], value_dict)
                for i, v in enumerate(value_dict):
                    text = v
                    assert text != ''
                    value_id2text[ds['name']][i] = text
                    value_id2tokenized_text[ds['name']][i] = tokenizer(text)['input_ids']
                    # print(text, tokenizer(text)['input_ids'])

            for i, value in enumerate(ontology_value_list):
                assert value != ''
                ontology_value_text2id[value] = i
                ontology_value_id2text[i] = value
                ontology_value_id2tokenized_text[i] = tokenizer(value)['input_ids']
            
            graph_meta =model.add_KB(
                value_id2tokenized_text,
                value_id2text,
                ds_list,
                ontology_value_list,
                ontology_value_text2id,
                ontology_value_id2text,
                ontology_value_id2tokenized_text,
            )


        if args.do_train:
            print("model training..")
            num_train_steps = int(len(train_dataloader) / args.batch_size * args.n_epochs)
            bert_params_ids = list(map(id, model.albert.parameters()))
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            enc_param_optimizer = list(model.named_parameters())

            enc_optimizer_grouped_parameters = [
                {'params': [p for n, p in enc_param_optimizer if id(p) in bert_params_ids and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.enc_lr},
                {'params': [p for n, p in enc_param_optimizer if id(p) in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.enc_lr},
                {'params': [p for n, p in enc_param_optimizer if id(p) not in bert_params_ids and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.base_lr},
                {'params': [p for n, p in enc_param_optimizer if id(p) not in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.base_lr}]

            enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr = args.base_lr)

            enc_scheduler = get_linear_schedule_with_warmup(enc_optimizer, num_warmup_steps=int(num_train_steps * args.enc_warmup), num_training_steps=num_train_steps)
            
            best_score = {'epoch': 0, 'overall_jga': 0, 'cate_jga': 0, 'noncate_jga': 0}
            # best_score = {'epoch': 0, 'gen_acc': 0, 'op_acc': 0, 'op_F1': 0}

            model.train()
            loss = 0
            generation_weight, extraction_weight = args.generation_weight, args.extraction_weight

            graph_dialogue = []
            dialog_history = []
            current_id = ""
            update_slot = {}
            turn_num = 0
            for epoch in range(args.n_epochs):
                print(f'model train in epoch [{epoch}/{args.n_epochs}]')
                batch_loss = []
                for step, batch in enumerate(train_dataloader):
                    b_input_ids, b_input_mask, b_slot_mask, b_segment_ids, b_state_position_ids, b_op_ids, b_pred_ops, \
                    b_gen_ids, b_slot_ans_ids, b_gold_ans_label, b_sid = batch

                    for data in range(len(b_sid)):
                        input_ids = b_input_ids[data].to(device)
                        input_mask = b_input_mask[data].to(device)
                        slot_mask = b_slot_mask[data].to(device)
                        segment_ids = b_segment_ids[data].to(device)
                        state_position_ids = b_state_position_ids[data].to(device)
                        op_ids = b_op_ids[data].to(device)
                        pred_ops = b_pred_ops[data].to(device)
                        slot_ans_ids = b_slot_ans_ids[data].to(device)

                        gen_ids = [b_gen_ids[data]]
                        sid = [b_sid[data]]

                        op = op_ids[0].cpu().detach().numpy().tolist()
                        sample_mask = (pred_ops.argmax(dim=-1) == 0) if turn == 1 else (op_ids == 0)

                        pre_forward_results = model.pre_forward(input_ids = input_ids,
                                                                token_type_ids = segment_ids,
                                                                state_positions = state_position_ids,
                                                                attention_mask = input_mask,
                                                                slot_mask = slot_mask)
                        dialog_node = pre_forward_results['dialog_node']
                        slot_node = pre_forward_results['slot_node']

                        if int(sid[0].split("_")[1]) == 0 or sid[0].split("_")[0] != current_id: 
                            current_id = sid[0].split("_")[0]
                            graph_dialogue = []
                            dialog_history = []
                            update_slot = {}
                            turn_num = 0
                            model.refresh_embeddings()
                        graph_dialogue.append(dialog_node)
                        input_id = input_ids.cpu().detach().numpy().tolist()
                        dialog_history.append(input_id)
                        
                        update_slot[int(turn_num)] = []
                        for op_idx, p in enumerate(op):
                            if p==0: update_slot[int(turn_num)].append(op_idx)
                        # print(f'turn{turn_num}')
                        # print(f'update_slot:{update_slot}')
                        turn_num += 1

                        graph_output_list = {}
                        for i, (op, mask) in enumerate(zip(op, cate_mask)):
                            if op==0: 
                                graph_output_list[i] = {}
                                graph_type='value' if mask else 'dialogue'
                                graph_forward_results = model.graph_forward(dial_embeddings=graph_dialogue,
                                                                            ds_embeddings=slot_node,
                                                                            graph_type=graph_type,
                                                                            update_slot=update_slot)
                                # graph_dial_output = graph_forward_results['dial_embeddings']
                                # graph_slot_output = graph_forward_results['ds_embeddings']
                                graph_atten = graph_forward_results['graph_attentions']
                                graph_output_list[i]['type'] = graph_type
                                # graph_output_list[i]['dial'] = graph_dial_output # list
                                # graph_output_list[i]['slot'] = graph_slot_output # list
                                graph_output_list[i]['atten'] = graph_atten[args.num_layer-1]
                        del dialog_node, slot_node
                        
                        inputs, start_logits, end_logits, _, gen_scores, _, _, _ = model.forward(input_ids = input_ids,
                                                                                                attention_mask = input_mask,
                                                                                                tokenizer = tokenizer,
                                                                                                graph_output_list = graph_output_list,
                                                                                                ontology_value_list = ontology_value_list,
                                                                                                dialog_history = dialog_history)
                        del graph_output_list

                        loss_classifier= masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                                        slot_ans_ids.contiguous(),
                                                                        sample_mask = sample_mask
                                                                        )
                        loss_extractor = compute_span_loss(gen_ids, inputs, start_logits, end_logits, sample_mask)
                        loss = loss_classifier + loss_extractor

                        loss.backward()
                        for name, par in model.named_parameters():
                            if par.requires_grad and par.grad is not None:
                                if torch.sum(torch.isnan(par.grad)) != 0:
                                    model.zero_grad()

                        batch_loss.append(loss.item())
                        torch.cuda.empty_cache()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        enc_optimizer.step()
                        enc_scheduler.step()
                        model.zero_grad()
                    if step % 100 == 0:
                        print(f'[{epoch}/{args.n_epochs}] [{step}/{len(train_dataloader)}] mean_loss :{np.mean(batch_loss)}')
                        print("Epoch {}\t Step {}\t loss {:.6f}\t time {}".format(epoch, step, loss, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        batch_loss = []

                graph_dialogue = []
                dialog_history = []
                torch.cuda.empty_cache()

                # save model per epoch
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.save_dir, 'checkpoint_epoch_' + str(epoch) + '.bin')
                torch.save(model_to_save.state_dict(), save_path)

                joint_acc, cate_acc, noncate_acc, gen_acc = evaluate(args, dev_dataloader, model, device, ans_vocab_nd, cate_mask, tokenizer=tokenizer, ontology=ontology)
                print(f'[evaluate] epoch:{epoch}\t loss:{loss}\t joint_acc:{joint_acc}\t cate_acc:{cate_acc}\t noncate_acc:{noncate_acc}\t gen_acc:{gen_acc}')
                    
                if joint_acc > best_score['overall_jga']:  
                    best_score['epoch'] = epoch
                    best_score['overall_jga'] =joint_acc
                    best_score['cate_jga'] = cate_acc
                    best_score['noncate_jga'] = noncate_acc
                    saved_name = 'model_best_turn_' + str(epoch) + '.bin'
                    save_path = os.path.join(args.save_dir, saved_name)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    params = {
                        'model': model_to_save.state_dict(),
                        'optimizer': enc_optimizer.state_dict(),
                        'scheduler': enc_scheduler.state_dict(),
                        'args': args
                    }
                    torch.save(params, save_path)
                    print("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}".format(epoch, joint_acc * 100, cate_acc * 100, noncate_acc * 100))
                print("[Best Score] overall_jga : %.3f, cate_jga : %.3f,noncate_jga : %.3f" % (best_score['overall_jga'], best_score['cate_jga'], best_score['noncate_jga']))
                print("\n")
                torch.cuda.empty_cache()
                model.train()

        else:
            model_evaluation(args, model, test_data_raw, tokenizer, slot_meta, ontology=ontology, ans_vocab=ans_vocab_nd, cate_mask=cate_mask)
            # joint_acc, cate_acc, noncate_acc, gen_acc = evaluate(args, dev_dataloader, model, device, ans_vocab_nd, cate_mask, tokenizer=tokenizer, ontology=ontology)



def evaluate(args, dev_dataloader, model, device, ans_vocab_nd, cate_mask, tokenizer=None, ontology=None):
    model.eval()
    start_predictions = []
    end_predictions = []
    has_ans_predictions = []
    has_ans_labels = []
    gen_predictions = []
    slot_ans_idx = []
    all_input_ids = []
    sample_ids = []
    gen_id_labels = []
    gold_ans_labels = []

    graph_dialogue = []
    dialog_history = []
    current_id = ""
    update_slot = {}
    turn_num = 0
    for step, batch in enumerate(tqdm(dev_dataloader)):
        b_input_ids, b_input_mask, b_slot_mask, b_segment_ids, b_state_position_ids, b_op_ids, b_pred_ops, \
        b_gen_ids, b_slot_ans_ids, b_gold_ans_label, b_sid = batch

        for data in range(len(b_sid)):
            input_ids = b_input_ids[data].to(device)
            input_mask = b_input_mask[data].to(device)
            slot_mask = b_slot_mask[data].to(device)
            segment_ids = b_segment_ids[data].to(device)
            state_position_ids = b_state_position_ids[data].to(device)
            op_ids = b_op_ids[data].to(device)
            slot_ans_ids = b_slot_ans_ids[data].to(device)

            gen_ids = [b_gen_ids[data]]
            gold_ans_label = [b_gold_ans_label[data]]
            sid = [b_sid[data]]
            # print(f'[{sid}]')

            op = op_ids[0].cpu().detach().numpy().tolist()

            pre_forward_results = model.pre_forward(input_ids = input_ids,
                                                    token_type_ids = segment_ids,
                                                    state_positions = state_position_ids,
                                                    attention_mask = input_mask,
                                                    slot_mask = slot_mask)
            dialog_node = pre_forward_results['dialog_node']
            slot_node = pre_forward_results['slot_node']

            if int(sid[0].split("_")[1]) == 0 or sid[0].split("_")[0] != current_id: 
                current_id = sid[0].split("_")[0]
                graph_dialogue = []
                dialog_history = []
                update_slot = {}
                turn_num = 0
                model.refresh_embeddings()
            
            graph_dialogue.append(dialog_node)
            input_id = input_ids.cpu().detach().numpy().tolist()
            dialog_history.append(input_id)
            
            update_slot_name = []
            update_slot[int(turn_num)] = []
            for op_idx, p in enumerate(op):
                if p==0: 
                    update_slot_name += [ontology[op_idx]['name']]
                    update_slot[int(turn_num)].append(op_idx)
            # print(f'update slot: {update_slot_name}')
            # print(f'update_slot:{update_slot}')
            turn_num += 1

            graph_output_list = {}
            for i, (op, mask) in enumerate(zip(op, cate_mask)):
                if op==0: 
                    graph_output_list[i] = {}
                    graph_type='value' if mask else 'dialogue'
                    graph_forward_results = model.graph_forward(dial_embeddings=graph_dialogue,
                                                                ds_embeddings=slot_node,
                                                                graph_type=graph_type,
                                                                update_slot=update_slot)
                    # graph_dial_output = graph_forward_results['dial_embeddings']
                    # graph_slot_output = graph_forward_results['ds_embeddings']
                    graph_atten = graph_forward_results['graph_attentions']
                    graph_output_list[i]['type'] = graph_type
                    # graph_output_list[i]['dial'] = graph_dial_output # list
                    # graph_output_list[i]['slot'] = graph_slot_output # list
                    graph_output_list[i]['atten'] = graph_atten[args.num_layer-1]
            
            del dialog_node, slot_node

            inputs, start_logits, end_logits, _, gen_scores, _, _, _ = model.forward(input_ids = input_ids,
                                                                                    attention_mask = input_mask,
                                                                                    tokenizer = tokenizer,
                                                                                    graph_output_list = graph_output_list,
                                                                                    ontology_value_list = ontology_value_list,
                                                                                    dialog_history = dialog_history)
            del graph_output_list

            start_predictions += start_logits.argmax(dim = -1).cpu().detach().numpy().tolist()
            end_predictions += end_logits.argmax(dim = -1).cpu().detach().numpy().tolist()
            # has_ans_predictions += pred_ops.argmax(dim = -1).view(-1).cpu().detach().numpy().tolist()
            has_ans_predictions += op_ids.cpu().detach().numpy().tolist() # gold op_labels for validation(same as train)
            gen_predictions += gen_scores.argmax(dim = -1).cpu().detach().numpy().tolist()
            has_ans_labels += op_ids.cpu().detach().numpy().tolist()
            slot_ans_idx += slot_ans_ids.cpu().detach().numpy().tolist()
            all_input_ids += inputs[:, 1:].cpu().detach().numpy().tolist()
            sample_ids += sid
            gen_id_labels += gen_ids
            gold_ans_labels += gold_ans_label
            
            del inputs, start_logits, end_logits, gen_scores
            torch.cuda.empty_cache()

    joint_acc, cate_acc, noncate_acc, gen_acc = joint_evaluation(start_predictions, 
                                                                end_predictions, 
                                                                gen_predictions, 
                                                                has_ans_predictions, 
                                                                slot_ans_idx, 
                                                                gen_id_labels, 
                                                                has_ans_labels, 
                                                                all_input_ids, 
                                                                ans_vocab_nd, 
                                                                gold_ans_labels, 
                                                                tokenizer=tokenizer, 
                                                                sid=sample_ids, 
                                                                catemask=cate_mask,
                                                                ontology=ontology)

    

    return joint_acc, cate_acc, noncate_acc, gen_acc


if __name__ ==  "__main__":
    main()
