import torch
import difflib
import time
from copy import deepcopy

from utils.data_utils import OP_SET
from utils.constant import n_slot, ontology_value_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def op_evaluation(start_prediction,end_prediction,gen_prediction,op_prediction,start_idx,end_idx,slot_ans_idx,\
                op_ids,input_ids,ans_vocab,sid=None,catemask=None):
    print("\nop_evaluation..")
    gen_guess=0.0
    gen_correct=0.0
    op_guess=0.0
    op_correct=0.0
    op_update_guess=0.0
    op_update_correct=0.0
    op_update_gold=0.0
    ans_pad_size=ans_vocab.shape[-1]
    ans_vocab=ans_vocab.tolist()
    sample_op=[]
    for i,op_pred in enumerate(op_prediction):
        sample_id = i // n_slot
        slot_id = i % n_slot
        op_guess+=1
        extract_ans = [2] + input_ids[sample_id][start_prediction[i]-1:end_prediction[i]] + [3]
        extract_ans += [0] * (ans_pad_size - len(extract_ans))
        
        isvalid=(extract_ans in ans_vocab[slot_id])
        
        sample_op+=[[1-op_pred,op_pred]]
        if op_ids[i]==0:
            op_update_gold+=1
        if op_pred == 0:
            op_update_guess += 1
        if op_pred==op_ids[i]:
            op_correct+=1
            if op_ids[i]==0:
                op_update_correct+=1
        if op_ids[i]==0:
            gen_guess+=1
            if catemask[slot_id]:
                if isvalid:
                # if start_idx[i]!=-1:
                    gen_correct+=1*(start_idx[i]==start_prediction[i])*(end_idx[i]==end_prediction[i])
                else:
                    gen_correct+=1*(gen_prediction[i]==slot_ans_idx[i])
            else:
                if (start_idx[i] == start_prediction[i]) and (end_idx[i] == end_prediction[i]):
                    gen_correct += 1
                elif start_idx[i]!=-1 and input_ids[sample_id][start_prediction[i]-1:end_prediction[i]]==input_ids[sample_id][start_idx[i]-1:end_idx[i]]:
                    gen_correct += 1
        if slot_id==29:
            sample_op=[]

    gen_acc=gen_correct/gen_guess if gen_guess!=0 else 0
    op_acc=op_correct/op_guess if op_guess!=0 else 0
    op_prec=op_update_correct/op_update_guess if op_update_guess!=0 else 0
    op_recall=op_update_correct/op_update_gold if op_update_gold!=0 else 0
    op_F1=2*(op_prec*op_recall)/(op_prec+op_recall) if op_prec+op_recall!=0 else 0
    print("op_update_correct: ", op_update_correct)
    print("op_update_gold: ", op_update_gold)
    print("op_update_guess: ", op_update_guess)
    print("gen_correct: ", gen_correct)
    print("gen_guess: ", gen_guess)
    print("gen_acc: ", gen_acc)
    print("Update score: operation precision: %.3f, operation_recall : %.3f,operation F1:%.3f"% (op_prec, op_recall,op_F1))
    return gen_acc, op_acc, op_prec, op_recall, op_F1

def joint_evaluation(start_prediction, end_prediction, gen_prediction, op_prediction, slot_ans_idx, gen_ids, op_ids, input_ids, 
                     ans_vocab, gold_ans_labels, tokenizer=None, sid=None, catemask=None, ontology=None):
    ans_pad_size = ans_vocab.shape[-1]
    ans_vocab = ans_vocab.tolist()
    gen_guess = 0.0
    gen_correct = 0.0
    joint_correct = 0.0
    catecorrect = 0.0
    noncatecorrect = 0.0
    cate_slot_correct = 0.0
    nocate_slot_correct = 0.0
    domain_joint = {"hotel": 0, "train": 0, "attraction": 0, "taxi": 0, "restaurant": 0}

    gen_acc = 0.0
    samples = 0.0
    joint_acc = 0.0
    cate_acc = 0.0
    noncate_acc = 0.0

    current_id = ""
    for i, op_pred_turn in enumerate(op_prediction):
        # print(f'[{sid[i]}]')
        # print(f'op_pred_turn:{op_pred_turn}')
        if int(sid[i].split('_')[-1]) == 0 or sid[i].split("_")[0] != current_id:
            current_id = sid[i].split("_")[0]
            last_state = [[] for k in op_pred_turn]
        current_state = [[] for k in op_pred_turn]
        gold_state = gold_ans_labels[i]
        # print(f'last_state__:{last_state}')
        # print(f'current_stat:{current_state}')
        # print(f'gold_state__:{gold_state}')
        iscate_correct = 1
        isnoncate_correct = 1
        domain_correct = {"hotel": 1, "train": 1, "attraction": 1, "taxi": 1, "restaurant": 1}
        for si, op_pred in enumerate(op_pred_turn):
            extract_ans = [2] + input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] + [3]
            extract_ans += [0] * (ans_pad_size - len(extract_ans))
            
            isvalid = (extract_ans in ans_vocab[si])

            if op_pred == 0:
                batch_correct = 0
                gen_guess += 1
                if catemask[si]:
                    if isvalid:
                        batch_correct += 1 * (((input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] + [30002]) in gen_ids[i][si]) 
                                            or (input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] == gen_ids[i][si]))
                        current_state[si] = input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] + [30002]
                        if ans_vocab[si].index(extract_ans) == slot_ans_idx[i][si]:
                            current_state[si] = gold_state[si]
                    else:
                        batch_correct += 1 * (gen_prediction[i][si] == slot_ans_idx[i][si])
                        current_state[si] = list(filter(lambda x: x not in [0, 2, 3], ans_vocab[si][gen_prediction[i][si]])) + [30002]
                        if gen_prediction[i][si] == slot_ans_idx[i][si]:
                            current_state[si] = gold_state[si]
                else:
                    batch_correct += 1 * ((input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] + [30002] in gen_ids[i][si]) 
                                          or (input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] == gen_ids[i][si]))
                    current_state[si] = input_ids[i][start_prediction[i][si] - 1:end_prediction[i][si]] + [30002]
                gen_correct += batch_correct
            else:
                current_state[si] = last_state[si]
        
        # print(f'current_stat:{current_state}')

        correct_mask = []
        for slot_id in range(len(current_state)):
            if current_state[slot_id] == 3 or (30000 in current_state[slot_id]):
                current_state[slot_id] = []
            if current_state[slot_id] == gold_state[slot_id]:
                correct_mask.append(1)
            else:
                if current_state[slot_id] != [] and gold_state[slot_id] != []:
                    sim = match(current_state[slot_id], gold_state[slot_id], tokenizer)
                    if sim > 0.9:
                        current_state[slot_id] = gold_state[slot_id]
                        correct_mask.append(1)
                    else:
                        correct_mask.append(0)
                else:
                    correct_mask.append(0)
            if correct_mask[-1] == 0:
                if catemask[slot_id]:
                    iscate_correct = 0
                else:
                    isnoncate_correct = 0
                name = ontology[slot_id]['name'].split("-")[0]
                domain_joint[name] = 0
            else:
                if catemask[slot_id]:
                    cate_slot_correct += 1
                else:
                    nocate_slot_correct += 1
        # print(f'correct_mask:{correct_mask}')

        correct = 1 if sum(correct_mask) == len(current_state) else 0
        joint_correct += correct
        catecorrect += iscate_correct
        noncatecorrect += isnoncate_correct
        for k in domain_joint.keys():
            domain_correct[k] += domain_correct[k]
        last_state = current_state

    gen_acc = gen_correct / gen_guess if gen_guess != 0 else 0
    samples = len(op_prediction)
    joint_acc = joint_correct/samples
    cate_acc = catecorrect/samples
    noncate_acc = noncatecorrect/samples

    print(f'\n')
    print(f'gen_correct:{gen_correct}')
    print(f'gen_guess:{gen_guess}')
    print(f'gen_acc:{gen_acc}')
    print(f'joint_acc:{joint_acc}')
    print(f'cate_acc:{cate_acc}')
    print(f'noncate_acc:{noncate_acc}')
    print(f'samples:{samples}')
    # print("Update score: operation precision: %.3f, operation_recall : %.3f,operation F1:%.3f" % (op_prec, op_recall, op_F1))
    return joint_acc, cate_acc, noncate_acc, gen_acc

def match(a, b, tokenizer):
    a = "".join(tokenizer.convert_ids_to_tokens(a))
    b = "".join(tokenizer.convert_ids_to_tokens(b))
    similarity = difflib.SequenceMatcher(None, a, b).quick_ratio()
    return similarity


def model_evaluation(args, model, test_data, tokenizer, slot_meta, ontology=None, ans_vocab=None, cate_mask=None,
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False):
    print("\nmodel_evaluation..")
    model.eval()

    last_slot_idx=[]
    last_dialog_state={}
    slot_idx=[]
    gen_guess = 0.0
    gen_correct = 0.0
    op_guess = 0.0
    op_correct = 0.0
    op_update_guess = 0.0
    op_update_correct = 0.0
    op_update_gold = 0.0
    joint_correct=0.0
    slot_correct=0.0
    cateslot=0.0
    nocateslot=0.0
    catecorrect=0.0
    noncatecorrect=0.0
    cate_slot_correct=0.0
    nocate_slot_correct=0.0
    domain_joint={"hotel":0,"train":0,"attraction":0,"taxi":0,"restaurant":0}
    domain_guess={"hotel":0,"train":0,"attraction":0,"taxi":0,"restaurant":0}
    # cate_mask=cate_mask.squeeze().cpu().detach().numpy().tolist()
    ans_pad_size = ans_vocab.shape[-1]
    ans_vocab = ans_vocab.tolist()
    domain_slot_correct = [0] * len(slot_meta)

    graph_dialogue = []
    dialog_history = []
    current_id = ""
    update_slot = {}
    turn_num = 0
    for di, i in enumerate(test_data):
        if i.turn_id == 0 or i.id.split("_")[0] != current_id:
            last_dialog_state={}
            for k,v in slot_meta.items():
                last_dialog_state[k]=[]
            last_slot_idx=[-1 if cate_mask[j] else [] for j in range(len(slot_meta))]
            last_ans_idx=[-1 if cate_mask[j] else [] for j in range(len(slot_meta))]
            current_id = i.id.split("_")[0]
            graph_dialogue = []
            dialog_history = []
            update_slot = {}
            turn_num = 0
            model.refresh_embeddings()

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.,turn=2,eval_token=True)
        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.,turn=2,eval_token=True)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.LongTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)
        slot_mask=torch.LongTensor([i.slot_mask]).to(device)
        gold_op_ids = torch.LongTensor([i.op_ids]).to(device)
        pred_op_ids=torch.LongTensor(i.pred_op.argmax(axis=-1))

        with torch.no_grad():
            op = pred_op_ids.cpu().detach().numpy().tolist()

            pre_forward_results = model.pre_forward(input_ids = input_ids,
                                                    token_type_ids = segment_ids,
                                                    state_positions = state_position_ids,
                                                    attention_mask = input_mask,
                                                    slot_mask = slot_mask)
            dialog_node = pre_forward_results['dialog_node']
            slot_node = pre_forward_results['slot_node']

            graph_dialogue.append(dialog_node)
            input_id = input_ids.cpu().detach().numpy().tolist()
            dialog_history.append(input_id)

            # print(f'[{i.id}]')
            update_slot_name = []
            update_slot[int(turn_num)] = []
            for op_idx, p in enumerate(op):
                if p==0: 
                    update_slot_name += [ontology[op_idx]['name']]
                    update_slot[int(turn_num)].append(op_idx)
            # print(f'update slot: {update_slot_name}')
            # print(f'update_slot: {update_slot}')
            turn_num += 1
            
            graph_output_list = {}
            for p, (op, mask) in enumerate(zip(op, cate_mask)):
                if op==0: 
                    graph_output_list[p] = {}
                    graph_type='value' if mask else 'dialogue'
                    graph_forward_results = model.graph_forward(dial_embeddings=graph_dialogue,
                                                                ds_embeddings=slot_node,
                                                                graph_type=graph_type,
                                                                update_slot=update_slot)
                    # graph_dial_output = graph_forward_results['dial_embeddings']
                    # graph_slot_output = graph_forward_results['ds_embeddings']
                    graph_atten = graph_forward_results['graph_attentions']
                    graph_output_list[p]['type'] = graph_type
                    # graph_output_list[p]['dial'] = graph_dial_output 
                    # graph_output_list[p]['slot'] = graph_slot_output 
                    graph_output_list[p]['atten'] = graph_atten[args.num_layer-1]
            
            del dialog_node, slot_node

            inputs, start_logits, end_logits, _, gen_scores, _, _, _ = model.forward(input_ids = input_ids,
                                                                            attention_mask = input_mask,
                                                                            tokenizer = tokenizer,
                                                                            graph_output_list = graph_output_list,
                                                                            ontology_value_list = ontology_value_list,
                                                                            dialog_history = dialog_history)
            del graph_output_list

        start_prediction = start_logits.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        end_prediction = end_logits.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        gen_predictions = gen_scores.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        #op_predictions = has_ans.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        op_predictions = pred_op_ids.view(-1).cpu().detach().numpy().tolist()
        gold_op_ids = gold_op_ids.view(-1).cpu().detach().numpy().tolist()
        slot_ans_idx = [i.slot_ans_ids[j] if gold_op_ids[j] == 0 else last_ans_idx[j] for j in range(len(op_predictions))]
        #slot_ans_idx = slot_ans_idx.view(-1).cpu().detach().numpy().tolist()

        torch.cuda.empty_cache()

        gen_ids=i.generate_ids
        start_idx, end_idx = find_value_idx(gen_ids, inputs, op_predictions)
        # start_idx=i.start_idx
        # end_idx=i.end_idx
 
        op_guess += 1
        slot_idx=[-1 if cate_mask[j] else [] for j in range(len(last_slot_idx))]
        iswrong=False
        iscatewrong=False
        isnocatewrong=False
        is_domain_correct={"hotel":True,"train":True,"attraction":True,"taxi":True,"restaurant":True}
        input_ids=inputs[0].cpu().detach().numpy().tolist()

        for idx,op_pred in enumerate(op_predictions):
            if op_pred==0:
                gen_guess+=1
            if cate_mask[idx]:
                extract_ans = [2] + input_ids[start_prediction[idx]-1:end_prediction[idx]] + [3]
                extract_ans += [0] * (ans_pad_size - len(extract_ans))
                isvalid = (extract_ans in ans_vocab[idx])
                if isvalid:
                    if op_pred==1:
                        slot_idx[idx]=last_slot_idx[idx]
                    else:
                        if start_prediction[idx]==start_idx[idx] and end_prediction[idx]==end_idx[idx]:
                            slot_idx[idx]=slot_ans_idx[idx]
                        else:
                            slot_idx[idx]=ans_vocab[idx].index(extract_ans)
                else:
                    if op_pred==1:
                        slot_idx[idx] = last_slot_idx[idx]
                    else:
                        if input_ids[start_prediction[idx]-1:end_prediction[idx]]==3:
                            slot_idx[idx]=-1
                        elif gen_predictions[idx]==len(ontology[idx]['db'])-2:
                            slot_idx[idx]=-1
                        else:
                            slot_idx[idx]=gen_predictions[idx]
            else:
                if op_pred == 1:
                    slot_idx[idx] = last_slot_idx[idx]
                else:
                    if start_prediction[idx]== start_idx[idx] and end_prediction[idx]== end_idx[idx]:
                        slot_idx[idx]=gen_ids[idx][0]
                        slot_ans_idx[idx]=gen_ids[idx][0]
                    elif start_prediction[idx]>=end_prediction[idx] or end_prediction[idx]-start_prediction[idx]>10:
                        slot_idx[idx] =[]
                    else:
                        slot_idx[idx]=[input_ids[start_prediction[idx]-1:end_prediction[idx]]+[30002]]

            if slot_ans_idx[idx] == len(ontology[idx]['db']) - 2:
                slot_ans_idx[idx]= -1
            if slot_idx[idx] == len(ontology[idx]['db']) - 2:
                slot_idx[idx] = -1
            if cate_mask[idx]:
                cateslot+=1
                if slot_idx[idx] != slot_ans_idx[idx]:
                    iswrong=True
                    iscatewrong=True
                    is_domain_correct[ontology[idx]['name'].split("-")[0]] = False
                else:
                    cate_slot_correct += 1
                    slot_correct+=1
                    domain_slot_correct[idx]+=1
                    if op_pred==0:
                        gen_correct+=1
            else:
                nocateslot+=1
                if slot_idx[idx]!=gen_ids[idx] and (slot_idx[idx]!=slot_ans_idx[idx]) and slot_idx[idx] not in gen_ids[idx] \
                    and (slot_idx[idx]!=3 or gen_ids!=[]) and (slot_idx[idx]==[] or gen_ids!=[3]):
                    iswrong = True
                    isnocatewrong=True
                    is_domain_correct[ontology[idx]['name'].split("-")[0]]=False
                else:
                    nocate_slot_correct += 1
                    domain_slot_correct[idx] += 1
                    slot_correct+=1
                    if op_pred==0:
                        gen_correct+=1
                if slot_idx[idx]==[3] or slot_idx[idx]==3:
                    slot_idx[idx]=[]


            if op_pred == gold_op_ids[idx]:
                op_correct += 1
                if gold_op_ids[idx] == 0:
                    op_update_correct += 1
            if gold_op_ids[idx] == 0:
                op_update_gold += 1
            if op_pred == 0:
                op_update_guess += 1

        if not iswrong:
            joint_correct+=1
        if not iscatewrong:
            catecorrect+=1
        if not isnocatewrong:
            noncatecorrect+=1
        v = is_domain_correct[i.turn_domain]
        domain_guess[i.turn_domain]+=1
        if v:
            domain_joint[i.turn_domain]+=1
        last_slot_idx=slot_idx
        last_ans_idx=slot_ans_idx
        for k,s in enumerate(last_slot_idx):
            if cate_mask[k]:
                if slot_idx[k]==-1:
                    last_dialog_state[ontology[k]['name']] = []
                else:
                    last_dialog_state[ontology[k]['name']]=ontology[k]['db'][s]
            else:
                if slot_idx[k]==[]:
                    last_dialog_state[ontology[k]['name']] = []
                else:
                    if isinstance(s[0],list):
                        s=s[0]
                    # if 30000 in s: s.remove(30000)
                    last_dialog_state[ontology[k]['name']] = tokenizer.convert_ids_to_tokens(s)
    domain_slot_acc={}
    for idx,d in enumerate(domain_slot_correct):
        domain_slot_acc[ontology[idx]['name']]=d/len(test_data)
    for ds in domain_slot_acc.items():
        print(ds)
    for dj,cor in domain_joint.items():
        print(dj)
        print(cor/domain_guess[dj])
    print("joint_acc: ", joint_correct/len(test_data))
    print("noncate_acc: ", noncatecorrect/len(test_data))
    print("cate_acc: ", catecorrect/len(test_data))
    print("gen_correct: ", gen_correct)
    print("gen_guess: ", gen_guess)
    print("gen_acc: ", gen_correct/gen_guess)
    print("slot_acc: ", slot_correct/(op_guess*30))
    print("cate_slot_acc: ", cate_slot_correct / cateslot)
    print("noncate_slot_acc: ", nocate_slot_correct/nocateslot)
    print("op_acc: ", op_correct/(op_guess*30))
    op_recall=op_update_correct/op_update_gold
    op_prec=op_update_correct/op_update_guess
    print("op_update_correct: ", op_update_correct)
    print("op_update_gold: ", op_update_gold)
    print("op_update_guess: ", op_update_guess)

    print("op_prec: ", op_prec)
    print("op_recall: ", op_recall)
    print("op_f1: ", 2*(op_prec*op_recall)/(op_prec+op_recall))

def find_value_idx(gen_ids, input_ids, sample_mask):
    # print(f'gen_ids:{gen_ids}')
    # print(f'input_ids:{input_ids.shape}') #torch.Size([1, 512])
    # print(f'sample_mask:{sample_mask}')
    start_idx = [-1 for i in range(n_slot)]
    end_idx = [-1 for i in range(n_slot)]
    batch_input = input_ids[0].cpu().detach().numpy().tolist()
    # print(f'batch_input:{batch_input}')
    for ti in range(n_slot):
        if sample_mask[ti]:
            continue
        value = gen_ids[ti][0] if gen_ids[ti]!=[] else [0]
        value = value[:-1] if isinstance(value, list) else [value]
        # print(f'value:{value}')
        
        for text_idx in range(len(input_ids[0]) - len(value)):
            if batch_input[text_idx: text_idx + len(value)] == value:
                start_idx[ti] = text_idx
                end_idx[ti] = text_idx + len(value) - 1
                break
    # print(f'start_idx:{start_idx}')
    # print(f'end_idx:{end_idx}')
    return start_idx, end_idx


if __name__ == "__main__":
    pass
