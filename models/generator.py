import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import AlbertModel
import torch.nn.functional as F
import math
from utils.constant import n_slot
from models.graph_model import GraphModel

class Generator(nn.Module):
    def __init__(self, args, ans_vocab, slot_mm, turn=2):
        super(Generator, self).__init__()
        bert_config = BertConfig.from_pretrained(args.model_name_or_path + "config.json")
        args.hidden_size = bert_config.hidden_size
        self.hidden_size = bert_config.hidden_size
        self.n_slot = n_slot
        self.args = args
        self.slot_mm = slot_mm
        self.turn = turn
        self.albert = AlbertModel.from_pretrained(args.model_name_or_path + "pytorch_model.bin",  config = bert_config)
        self.albert.resize_token_embeddings(args.vocab_size)
        self.input_drop = nn.Dropout(p = 0.5)
        smask = ans_vocab.sum(dim = -1).eq(0).long()
        smask = slot_mm.long().mm(smask)
        self.slot_mm = nn.Parameter(slot_mm, requires_grad = False)
        self.slot_ans_mask = nn.Parameter(smask, requires_grad = False)
        self.ans_vocab = nn.Parameter(torch.FloatTensor(ans_vocab.size(0),  ans_vocab.size(1),  self.hidden_size), requires_grad = True)
        self.max_ans_size = ans_vocab.size(-1)
        self.slot_ans_size = ans_vocab.size(1)
        self.eslots = ans_vocab.size(0)
        self.ans_bias = nn.Parameter(torch.FloatTensor(ans_vocab.size(0),  ans_vocab.size(1),  1), requires_grad = True)
        if turn!=2:
            self.has_ans1 = nn.Linear(self.hidden_size, 2)
        self.start_output = nn.Linear(self.hidden_size, self.hidden_size)
        self.end_output = nn.Linear(self.hidden_size, self.hidden_size)

        self.dial_node = nn.Linear(self.hidden_size, 1)
        self.slot_node = nn.Linear(self.hidden_size, 1)
        self.maxseq = args.max_seq_length

        # add non-infer classifier
        torch.nn.init.xavier_normal_(self.ans_bias)
        torch.nn.init.xavier_normal_(self.ans_vocab)
        self.layernorm = torch.nn.LayerNorm(self.hidden_size)
        # self.albert = torch.nn.DataParallel(self.albert, device_ids = [0, 1])

        # Graph
        self.graph_model = GraphModel(self.args)
        self.graph_mode = args.graph_mode
        self.graph_residual = args.graph_residual
        self.cls_loss = args.cls_loss
        self.connect_type = args.connect_type
        self.device = args.device
        # - refresh_embeddings
        self.n_embd = self.hidden_size
        self.embedding_layer = nn.Embedding(args.vocab_size, self.n_embd)

    def pre_forward(self, input_ids, token_type_ids, state_positions, attention_mask, slot_mask):
        enc_outputs = self.albert(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
        sequence_output, pooled_output = enc_outputs[:2]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)

        return dict(
            dialog_node=pooled_output,
            slot_node=state_output
        )
    
    def graph_forward(self, dial_embeddings, ds_embeddings, graph_type, update_slot):
        self.graph_type = graph_type
        # print(f'graph_type: {graph_type}')

        ts_ds_embeddings = ds_embeddings.permute(0, 2, 1)  # B x N x F -> B x F x N
        ts_dial_embeddings = torch.stack(dial_embeddings)
        ts_dial_embeddings = ts_dial_embeddings.permute(1, 2, 0) # N x B x F -> B x F x N

        E = 1
        B, _, N_ds = ts_ds_embeddings.size()
        _, _, N_dial = ts_dial_embeddings.size()

        if graph_type == 'value':
            S = self.S
            N_sv, _ = S.size()
            self.N_sv = N_sv
            
            d_row = torch.zeros(N_dial, N_sv)
            d_col = torch.zeros(N_sv+N_dial, N_dial)
            
            # connect all slot to dialg
            for idx in range(N_ds):
                d_row[:, idx] = 1
                d_col[idx, :] = 1
                
            S = torch.cat([S, d_row], dim=0)
            S = torch.cat([S, d_col], dim=1)
            del d_row, d_col
            
            S = S.repeat(B, E, 1, 1).to(ts_ds_embeddings.device)
            self.graph_model.add_GSO(S)

            ontology_value_embeddings = self.ontology_value_embeddings.permute(1, 0).repeat(B, 1, 1).to(ts_ds_embeddings.device)
            merged_embeddings = torch.cat([ts_ds_embeddings, ontology_value_embeddings, ts_dial_embeddings], dim=-1)
            del ts_ds_embeddings, ontology_value_embeddings, ts_dial_embeddings
            
            ts_merged_embeddings_output = self.graph_model(merged_embeddings)

            if self.graph_residual == True:
                # Residual Connection
                ts_merged_embeddings_output += merged_embeddings
                
            # Get attentions
            attentions = self.graph_model.get_GSO()
            self.graph_attentions = attentions


        elif graph_type == 'dialogue':
            N = N_ds+N_dial
            S = torch.zeros(N, N)

            # connect each dialogue and updated slot
            for ti, slot_id in update_slot.items():
                for si in slot_id:
                    # print(f'{ti+N_ds} and {si} connected!')
                    S[ti+N_ds,si]=1
                    # print(f'{si} and {ti+N_ds} connected!')
                    S[si,ti+N_ds]=1
            S[-1, :N_ds] = 0
            S[:N_ds, -1] = 0

            #connect current dial to history
            S[N_ds+N_dial-1:, N_ds:] = 1
            S[N_ds:, N_ds+N_dial-1:] = 1
            S[-1,-1] = 0
            
            #connect dialogue sequence
            for idx in range(N_dial-1):
                S[N_ds+idx+1, N_ds+idx]=1
            
            S = S.repeat(B, E, 1, 1).to(ts_ds_embeddings.device)
            self.graph_model.add_GSO(S)

            merged_embeddings = torch.cat([ts_ds_embeddings, ts_dial_embeddings], dim=-1)
            del ts_ds_embeddings, ts_dial_embeddings
            
            ts_merged_embeddings_output = self.graph_model(merged_embeddings)

            if self.graph_residual == True:
                # Residual Connection
                ts_merged_embeddings_output += merged_embeddings
            
            # Get attentions
            attentions = self.graph_model.get_GSO()
            self.graph_attentions = attentions


        loss = None
        logits = []
        del ts_merged_embeddings_output

        # return dict(
        #     dial_embeddings = dial_embeddings_output,
        #     ds_embeddings=ds_embeddings_output,
        #     loss=loss,
        #     logits=logits,
        #     graph_attentions=attentions
        # )
        return dict(
            graph_attentions=attentions
        )
    
    def forward(self, input_ids, attention_mask, tokenizer, graph_output_list, ontology_value_list, dialog_history):
        inputs = input_ids.cpu().detach().numpy().tolist()
        attention_mask = attention_mask.cpu().detach().numpy().tolist()
        pad_token_idx = attention_mask[0].index(0) if 0 in attention_mask[0] else None
        inputs = inputs[0][:pad_token_idx]

        dial_arg_list =[]
        for p, graph_output in graph_output_list.items():
            if graph_output['type'] == 'value':
                # continue
                atten = graph_output['atten'][0][0]
                slot_atten = atten[p][:self.N_sv]
                max_score = np.max(slot_atten)
                argmax = np.argmax(slot_atten)
                v = ['[CLS]']+tokenizer.tokenize(ontology_value_list[argmax-self.n_slot])+['[EOS]']
                value = tokenizer.convert_tokens_to_ids(v)
                inputs = inputs+value

            elif graph_output['type'] == 'dialogue':
                atten = graph_output['atten'][0][0]
                dial_atten = atten[-1]
                max_score = np.max(dial_atten)
                argmax = np.argmax(dial_atten)
                if max_score!=0: argmax= argmax-self.n_slot
                else: continue
                if argmax in dial_arg_list: continue
                dial_arg_list.append(argmax)
                d = dialog_history[argmax]
                sep_token_idx = d[0].index(3)
                d = d[0][1:sep_token_idx+1]
                inputs = d+inputs
        
        if len(inputs)>self.maxseq: inputs=inputs[-self.maxseq+1:] 
        assert len(inputs)<=self.maxseq-1

        slot_token=30000
        slot_position = []
        for i, t in enumerate(inputs):
            if t == slot_token:
                slot_position.append(i)
        state_positions = torch.LongTensor([slot_position]).to(self.device)

        slot_token_idx = inputs.index(30000)
        tmp_dial = inputs[:slot_token_idx]
        tmp_state = inputs[slot_token_idx:]

        cls_token_idx = tmp_dial.index(2)
        dial1 = [2]+tmp_dial[:cls_token_idx] # history
        dial2 = tmp_dial[cls_token_idx:] # current
        diag = dial1+dial2

        inputs = [2]+inputs

        segment = [0] * len(dial1) + [1] * len(dial2)
        segment = segment + [1]*len(tmp_state) 
        input_mask = [1] * len(inputs) 
        slot_mask = [1] * len(diag) 

        # new padding
        inputs = inputs + [0] * (self.maxseq-len(input_mask))
        segment_ids = segment + [0] * (self.maxseq-len(input_mask))
        input_mask = input_mask + [0] * (self.maxseq-len(input_mask))
        slot_mask = slot_mask + [0] * (self.maxseq-len(slot_mask))

        inputs = torch.LongTensor([inputs]).to(self.device)
        segment_ids = torch.LongTensor([segment_ids]).to(self.device)
        input_mask = torch.LongTensor([input_mask]).to(self.device)
        slot_mask = torch.LongTensor([slot_mask]).to(self.device)

        enc_outputs = self.albert(input_ids=inputs,
                                token_type_ids=segment_ids,
                                attention_mask=input_mask)
        sequence_output, pooled_output = enc_outputs[:2]

        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        sequence_output=self.input_drop(sequence_output)
        seq_len=sequence_output.size(1)


        state_output=state_output.view(-1,1,self.hidden_size)
        start_output=self.start_output(sequence_output)
        end_output=self.end_output(sequence_output)
        start_output=self.layernorm(start_output)
        end_output=self.layernorm(end_output)
        start_atten_m = state_output.view(-1,self.n_slot,self.hidden_size).bmm(start_output.transpose(-1,-2)).view(-1,self.n_slot,seq_len)/math.sqrt(self.hidden_size)
        end_atten_m = state_output.view(-1,self.n_slot,self.hidden_size).bmm(end_output.transpose(-1,-2)).view(-1,self.n_slot,seq_len)/math.sqrt(self.hidden_size)
        start_logits =start_atten_m.masked_fill(slot_mask.unsqueeze(1)==0,-1e9)
        end_logits = end_atten_m.masked_fill(slot_mask.unsqueeze(1)==0,-1e9)
        if self.turn==2:
            start_logits_softmax =F.softmax(start_logits[:,:,1:],dim=-1)
            end_logits_softmax =F.softmax(end_logits[:,:,1:],dim=-1)
        else:
            start_logits_softmax = F.softmax(start_logits, dim=-1)
            end_logits_softmax = F.softmax(end_logits, dim=-1)
        
        
        ques_attn=F.softmax((sequence_output.repeat(self.n_slot,1,1).bmm(state_output.transpose(-1,-2))/math.sqrt(self.hidden_size)).masked_fill(slot_mask.repeat(self.n_slot,1).unsqueeze(-1)==0,-1e9),dim=1)
        sequence_pool_output=ques_attn.transpose(-1,-2).bmm(sequence_output.repeat(self.n_slot,1,1)).squeeze()
        if self.turn==2:
            has_ans=torch.Tensor([1]).cuda()
        else:
            has_ans=self.has_ans1(sequence_pool_output).view(-1,self.n_slot,2)

        #category answer generating
        sequence_pool_output=sequence_pool_output.view(-1,self.n_slot,self.hidden_size)
        category_ans=sequence_pool_output.transpose(0,1).bmm(self.slot_mm.mm(self.ans_vocab.view(self.eslots,-1)).view(self.n_slot,self.slot_ans_size,-1).transpose(-1,-2))+self.slot_mm.mm(self.ans_bias.squeeze()).unsqueeze(1)
        category_ans=category_ans.transpose(0,1)
        category_ans=category_ans.masked_fill((self.slot_ans_mask==1).unsqueeze(0),-1e9)
        category_ans_softmax=F.softmax(category_ans,dim=-1)

        return inputs, start_logits_softmax, end_logits_softmax, has_ans, category_ans_softmax, start_logits, end_logits, category_ans
    
    def refresh_embeddings(self):
        '''
        Refresh value candidate embeddings using the wte layer
        wte = nn.Embedding(config.vocab_size, config.n_embd)
        n_embd=768
        n_embd: Dimensionality of the embeddings and hidden states.
        '''
        
        self.ontology_value_embeddings = torch.zeros((len(self.ontology_value_list), self.n_embd)).to(self.device)
        for index, text in self.ontology_value_id2tokenized_text.items():
            ids = torch.LongTensor(text).to(self.device)
            embeddings = self.embedding_layer(ids)
            agg_embeddings = torch.sum(embeddings, dim=0)
            self.ontology_value_embeddings[index] = agg_embeddings

        self.ontology_value_embeddings = self.ontology_value_embeddings.detach()

    def add_KB(self,
               value_id2tokenized_text,
               value_id2text,
               ds_list,
               ontology_value_list,
               ontology_value_text2id,
               ontology_value_id2text,
               ontology_value_id2tokenized_text,
               ):
        """Add KB data to the model

        Args:
            value_id2tokenized_text (Dict): {str_ds_pair: {0: [id1 id2 id3], 1: ...}}
            value_id2text (Dict): {str_ds_pair: {0: 'none', 1: ...}}
            ds_list (List): list of ds pairs
            ontology_value_list (List): all values
            ontology_value_text2id (Dict):  {'none': 0, 'dont care':1,...}
            ontology_value_id2text (Dict): {0: 'none', 1: 'dont care',...}
            ontology_value_id2tokenized_text (Dict): {0: [id1 id2 id3], 1: ...}
        """
        self.value_id2tokenized_text = value_id2tokenized_text
        self.value_id2text = value_id2text
        self.ds_list = ds_list
        self.ontology_value_list = ontology_value_list
        self.ontology_value_text2id = ontology_value_text2id
        self.ontology_value_id2text = ontology_value_id2text
        self.ontology_value_id2tokenized_text = ontology_value_id2tokenized_text
        self.mapping_class_indices_to_ontology = {}
        for ds_index, str_ds_pair in enumerate(ds_list):
            all_indices_of_values = []
            for id in range(len(value_id2text[str_ds_pair])):
                text = value_id2text[str_ds_pair][id]
                all_indices_of_values.append(ontology_value_text2id[text])
            self.mapping_class_indices_to_ontology[ds_index] = all_indices_of_values

        self.refresh_embeddings()

        if self.graph_mode == 'full':
            N_ds = len(self.ds_list)
            N_v = len(self.ontology_value_list)
            N = N_ds + N_v
            S = torch.zeros((N, N))

            for ds_index, str_ds_pair in enumerate(self.ds_list):
                value_dict = self.value_id2text[str_ds_pair]
                for i in range(len(value_dict)):
                    # print('ds {} item {}: {}'.format(str_ds_pair, i, self.value_id2text[str_ds_pair][i]))
                    index_in_ontology = self.ontology_value_text2id[self.value_id2text[str_ds_pair][i]]
                    # print('corresponding index in ontology:', index_in_ontology)

                    # i: target, j: src
                    # connect ds node to value node
                    if self.connect_type == 'ds_value_only':
                        S[index_in_ontology + N_ds, ds_index] = 1
                    else:
                        # allow all ds nodes to pass features to this value node
                        S[index_in_ontology + N_ds, :N_ds] = 1

                    # Connect value node to ds node
                    S[ds_index, index_in_ontology + N_ds] = 1
                    # print('{} and {} connected'.format(index_in_ontology + N_ds, ds_index))

            self.S = S
        
        return S


