from transformers import CLIPTextModel  # text encoder
from transformers import CLIPVisionModel # image encoder
from transformers import CLIPConfig   # configuration
from transformers import CLIPModel
import torch.nn as nn
import math
import torch

class CrossAdaptertT2I(nn.Module):
    
    def __init__(self, text_config, vision_config):
        super(CrossAdaptertT2I, self).__init__()
        self.text_dim, self.embed_dim = text_config.hidden_size, vision_config.hidden_size
        self.num_heads = vision_config.num_attention_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = vision_config.attention_dropout
        self.layernorm = nn.LayerNorm(self.embed_dim)
    
    def _shape(self, key, bsz, num_head, seq_len, head_dim):
        return key.reshape(bsz, seq_len, num_head, head_dim).transpose(1, 2).contiguous()
    
    def forward(self, vision_query, text_key, text_value, attention_mask):
        # [128, 98, 768], [128, 77, 512], [128, 77, 512], [bsz*num_head, tgt_len, src_len]
        src_len, tgt_len = text_key.size(1), vision_query.size(1)
        bsz, num_head, head_dim = vision_query.size(0), self.num_heads, self.head_dim
        
        query_states = self.q_proj(vision_query) * self.scale
        key_states = self.k_proj(text_key)
        value_states = self.v_proj(text_value)
        
        src_proj_shape = (bsz * num_head, src_len, head_dim)
        tgt_proj_shape = (bsz * num_head, tgt_len, head_dim)
        
        query_states = self._shape(query_states, bsz, num_head, tgt_len, head_dim).view(*tgt_proj_shape)
        key_states = self._shape(key_states, bsz, num_head, src_len, head_dim).view(*src_proj_shape)
        value_states = self._shape(value_states, bsz, num_head, src_len, head_dim).view(*src_proj_shape)
        
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout)
        attn_output = torch.bmm(attn_probs, value_states)
        
        attn_output = attn_output.view(bsz, num_head, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)
        
        attn_output = self.o_proj(attn_output)
        
        # 返回残差和layernormd的结果
        return self.layernorm(attn_output + vision_query)


class CrossAdaptertI2T(nn.Module):
    
    def __init__(self, text_config, vision_config):
        super(CrossAdaptertI2T, self).__init__()
        self.vision_dim, self.embed_dim = vision_config.hidden_size, text_config.hidden_size
        self.num_heads = text_config.num_attention_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.vision_dim,self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = text_config.attention_dropout
        self.layernorm = nn.LayerNorm(self.embed_dim)
    
    def _shape(self, key, bsz, num_head, seq_len, head_dim):
        return key.reshape(bsz, seq_len, num_head, head_dim).transpose(1, 2).contiguous()
    
    def forward(self, text_query, vision_key, vision_value, attention_mask):
        # [128, 77, 512], [128, 98, 768], [128, 98, 768], [128, 1, 77, 98]
        src_len, tgt_len = vision_key.size(1), text_query.size(1)
        bsz, num_head, head_dim = text_query.size(0), self.num_heads, self.head_dim
        # [128, 77, 512], [128, 98, 512], [128, 98, 512]
        query_states = self.q_proj(text_query) * self.scale
        key_states = self.k_proj(vision_key)
        value_states = self.v_proj(vision_value)
        
        src_proj_shape = (bsz * num_head, src_len, head_dim)
        tgt_proj_shape = (bsz * num_head, tgt_len, head_dim)
        
        query_states = self._shape(query_states, bsz, num_head, tgt_len, head_dim).view(*tgt_proj_shape)
        key_states = self._shape(key_states, bsz, num_head, src_len, head_dim).view(*src_proj_shape)
        value_states = self._shape(value_states, bsz, num_head, src_len, head_dim).view(*src_proj_shape)
        
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout)
        attn_output = torch.bmm(attn_probs, value_states)
        
        attn_output = attn_output.view(bsz, num_head, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)
        
        attn_output = self.o_proj(attn_output)
        
        # 返回残差和layernormd的结果
        return self.layernorm(attn_output + text_query)

        
class ClipSimAdaptive(nn.Module):
    
    def __init__(self, freezing=True) -> None:
        super(ClipSimAdaptive, self).__init__()
        # basic setting for the multimodal encoder
        clip_pretrained_path = 'clip-vit-base-patch32'
        clip_config = CLIPConfig()
        self.text_encoder_config = clip_config.text_config
        self.vision_encoder_config = clip_config.vision_config
        original_clip = CLIPModel(clip_config)
        original_clip = original_clip.from_pretrained(clip_pretrained_path)
        self.text_encoder = original_clip.text_model
        self.vision_encoder = original_clip.vision_model
        self.vision_projection = original_clip.visual_projection
        self.text_projection = original_clip.text_projection
        self.logit_scale = original_clip.logit_scale
        self.alpha = nn.Parameter(torch.tensor(1/3))
        self.beta = nn.Parameter(torch.tensor(1/3))
        del original_clip

        # freezing the original clip and execute the bitfit
        trainable_component = ['norm', 'bias']

        for name, parameter in self.text_encoder.named_parameters():
            if any([trainable_component_ele in name for trainable_component_ele in trainable_component]):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = False
        for name, parameter in self.vision_encoder.named_parameters():
            if any([trainable_component_ele in name for trainable_component_ele in trainable_component]):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = False
        insert_index_text = [1, 3, 9]

        for i, index in enumerate(insert_index_text):
            self.text_encoder.encoder.layers.insert(index+1, CrossAdaptertI2T(self.text_encoder_config, self.vision_encoder_config))
            for j in range(i+1, len(insert_index_text)):
                insert_index_text[j] = insert_index_text[j] + 1
        
        insert_index_vision = [1, 3, 9]

        for i, index in enumerate(insert_index_vision):
            self.vision_encoder.encoder.layers.insert(index+1, CrossAdaptertT2I(self.text_encoder_config, self.vision_encoder_config))
            for j in range(i+1, len(insert_index_vision)):
                insert_index_vision[j] = insert_index_vision[j] + 1
        
        assert insert_index_text == insert_index_vision
        self.adapter_index = [insert_index_vision[i] + 1 for i in range(len(insert_index_vision))] #[3, 7, 11, 15]
        
        # set the position embedding for aux image
        self.aux_position_embedding = nn.Embedding(48, self.vision_encoder_config.hidden_size)
        self.register_buffer("aux_position_ids", torch.arange(48).reshape(1, -1))

        # set the token type id embedding for vision encoder
        vision_token_type = 2
        self.vision_token_type_embedding = nn.Embedding(vision_token_type, self.vision_encoder_config.hidden_size)
        self.register_buffer("vision_token_type_ids", torch.cat([torch.zeros((1, 50), dtype=torch.long), torch.ones((1, 48), dtype=torch.long)], dim=1))
    
    def _build_text_causal_attention_mask(self, text_seq_len):
        mask = torch.empty(text_seq_len, text_seq_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    def _build_image_to_text_attention_maks(self, text_seq_len, vision_seq_len):
        mask = torch.zeros(1, text_seq_len, vision_seq_len)
        return mask

    def _prepare_attention_mask(self, attention_mask):
        inv_attention_mask = 1.0 - attention_mask
        return inv_attention_mask.masked_fill(inv_attention_mask.bool(), torch.finfo(torch.float32).min)
    
    def _many_to_many_sim_compute(self, text_factor, vision_factor, attention_mask=None):
        if text_factor.size(1) == 1:

            mean_text_factor = text_factor.mean(dim=1)
            mean_vision_factor = vision_factor.mean(dim=1)
            nor_text_factor = mean_text_factor / mean_text_factor.norm(dim=1, keepdim=True)
            nor_vision_factor = mean_vision_factor / mean_vision_factor.norm(dim=1, keepdim=True)
            pos_sim = torch.diag(torch.mm(nor_text_factor, nor_vision_factor.T))
            pos_sim = (pos_sim + 1) / 2
            return pos_sim
        else:
            # score = torch.bmm(text_factor, vision_factor.transpose(1, 2)) / math.sqrt(text_factor.size(-1))
            score = torch.bmm(text_factor, vision_factor.transpose(1, 2))  #[1024,77,49]
     
            attention_mask[:, 0, :] = torch.finfo(torch.float32).min
            attention_mask[torch.arange(len(attention_mask)), (attention_mask>=0).sum(dim=1).squeeze(), :] = torch.finfo(torch.float32).min  #(attention_mask>=0).sum(dim=1): [1024,1]每个样本atten mask的有效位置的个数
            if attention_mask is not None:
                score = score + attention_mask
                pos_score_mask = (attention_mask >= 0).squeeze(-1)
            pos_score = nn.functional.softmax(score, dim=-1)
            pos_score_used_tokens = pos_score_mask.sum(dim=1)
          
            pos_sim = (torch.max(pos_score, dim=2).values) * pos_score_mask
            pos_sim = torch.sum(pos_sim, dim=1) / pos_score_used_tokens
            
            neg_score = nn.functional.softmax(score, dim=1)
            neg_sim = torch.mean(torch.max(neg_score, dim=1).values, dim=1)
            return 0.5 * (pos_sim + neg_sim)
    
    def _many_to_many_sim_compute_try(self, phrase_embedding, aux_embedding, text_embedding, vision_embedding, attention_mask=None):
        
        phrase_to_vision_patch_scores = torch.bmm(phrase_embedding, vision_embedding.transpose(1, 2)) / math.sqrt(phrase_embedding.size(-1))
        # phrase_to_vision_patch_scores = torch.bmm(phrase_embedding, vision_embedding.transpose(1, 2))
        phrase_to_vision_patch_scores = nn.functional.softmax(phrase_to_vision_patch_scores, dim=-1)
        phrase_to_vision_patch_sim = torch.max(phrase_to_vision_patch_scores, dim=-1).values.squeeze()
        
        aux_to_text_token_scores = torch.bmm(aux_embedding, text_embedding.transpose(1, 2)) / math.sqrt(phrase_embedding.size(-1))
        if attention_mask is not None:
            aux_to_text_token_scores = aux_to_text_token_scores + attention_mask
        aux_to_text_token_scores = nn.functional.softmax(aux_to_text_token_scores, dim=-1)
        aux_to_text_token_sim = torch.mean(torch.max(aux_to_text_token_scores, dim=-1).values, dim=1)
        
        return (phrase_to_vision_patch_sim + aux_to_text_token_sim) / 2
        # return phrase_to_vision_patch_sim, aux_to_text_token_sim
        
    def _many_to_many_sim_compute_try2(self, phrase_embedding, aux_embedding):
        phrase_to_aux_sim = torch.zeros(aux_embedding.size(0)).cuda() #[1024]
        aux_to_phrase_sim = torch.zeros(aux_embedding.size(0)).cuda()
        for i in range(len(phrase_embedding)): #len(phrase_embedding)=1024
            sample_phrase_embedding = phrase_embedding[i]   #batch里每个样本的phrase的维度:[num_phrase, 512]
            sample_aux_embedding = aux_embedding[i]  #[3,512]
            # sample_phrase_to_aux_scores = torch.mm(sample_phrase_embedding, sample_aux_embedding.T) / math.sqrt(aux_embedding.size(-1))
            sample_phrase_to_aux_scores = torch.mm(sample_phrase_embedding, sample_aux_embedding.T) #[num_phrase,3]
            sample_phrase_to_aux_sim = nn.functional.softmax(sample_phrase_to_aux_scores, dim = -1)#[num_phrase,3]
            phrase_to_aux_sim[i] = torch.mean(torch.max(sample_phrase_to_aux_sim, dim=-1).values)#[5]->[1]
            
            # sampler_aux_to_phrase_scores = torch.mm(sample_aux_embedding, sample_phrase_embedding.T) / math.sqrt(aux_embedding.size(-1))
            sampler_aux_to_phrase_scores = torch.mm(sample_aux_embedding, sample_phrase_embedding.T) #[3,5 ]
            sample_aux_to_phrase_sim = nn.functional.softmax(sampler_aux_to_phrase_scores, dim = -1)
            aux_to_phrase_sim[i] = torch.mean(torch.max(sample_aux_to_phrase_sim, dim=-1).values)
        return (phrase_to_aux_sim + aux_to_phrase_sim) / 2
            
    # compute the similarity and return the whole results   
    def forward(self, input_id, attention_mask_org, phrase_position, img, aux_img, data_id):

        bsz = input_id.size(0)
        # get the text embedding [bsz, 77, 512]
        text_embedding = self.text_encoder.embeddings(input_id)
        # get the vision embedding [bsz, 50, 768]
        org_vision_embedding = self.vision_encoder.embeddings(img) 
        # for the visual grounding image [bsz, 48, 768]
        aux_vision_embedding = self.vision_encoder.embeddings.patch_embedding(aux_img.to(dtype=self.vision_encoder.embeddings.patch_embedding.weight.dtype))
        aux_vision_embedding = aux_vision_embedding.flatten(2).transpose(1, 2)
        aux_vision_embedding = aux_vision_embedding.reshape(bsz, -1, self.vision_encoder_config.hidden_size)
        aux_vision_embedding = aux_vision_embedding + self.aux_position_embedding(self.aux_position_ids)
        # [bsz, 98, 768]
        vision_embedding = torch.cat([org_vision_embedding, aux_vision_embedding], dim=1)  
        # if find the type embedding is no use, delete it
        vision_embedding = vision_embedding + self.vision_token_type_embedding(self.vision_token_type_ids)
        vision_embedding = self.vision_encoder.pre_layrnorm(vision_embedding)
        
        # preprare the causal attention mask and attention mask for text encoder layer
        text_causal_attention_mask = self._build_text_causal_attention_mask(text_embedding.size(1))
        if torch.cuda.is_available():
            text_causal_attention_mask = text_causal_attention_mask.reshape(1, 1, *text_causal_attention_mask.shape).expand(bsz, -1, -1, -1).cuda()
        attention_mask = self._prepare_attention_mask(attention_mask_org)
        text_attention_mask = attention_mask.reshape(bsz, 1, 1, attention_mask.size(-1)).expand(-1, -1, attention_mask.size(-1), -1)
        
        # prepare the cross attention mask for the adapter
        text_to_image_attention_mask = attention_mask.reshape(bsz, 1, 1, attention_mask.size(-1)).expand(-1, self.vision_encoder_config.num_attention_heads, vision_embedding.size(1), -1)
        text_to_image_attention_mask = text_to_image_attention_mask.reshape(bsz*self.vision_encoder_config.num_attention_heads, vision_embedding.size(1), text_embedding.size(1))
        image_to_text_attention_mask = self._build_image_to_text_attention_maks(text_embedding.size(1), vision_embedding.size(1))
        if torch.cuda.is_available():
            image_to_text_attention_mask = image_to_text_attention_mask.cuda()

        for index, (text_layer, vision_layer) in enumerate(zip(self.text_encoder.encoder.layers, self.vision_encoder.encoder.layers)):
            if index in self.adapter_index:
                text_embedding = text_layer(text_embedding, vision_embedding, vision_embedding, image_to_text_attention_mask)
                vision_embedding = vision_layer(vision_embedding, text_embedding, text_embedding, text_to_image_attention_mask)
                
                # for brief, choosing the next scheme，and change the cross attention to self attention
                # text_embedding = text_layer(text_embedding, vision_embedding[:, 0, :].unsqueeze(1), vision_embedding[:, 0, :].unsqueeze(1), image_to_text_attention_mask[:, :, :, 0].unsqueeze(-1))
                # vision_embedding = vision_layer(vision_embedding, text_embedding[:, 0, :].unsqueeze(1), text_embedding[:, 0, :].unsqueeze(1), text_to_image_attention_mask[:, :, :, 0].unsqueeze(-1))
            else:
                text_embedding = text_layer(hidden_states=text_embedding, attention_mask=text_attention_mask, causal_attention_mask=text_causal_attention_mask)[0]
                vision_embedding = vision_layer(hidden_states=vision_embedding, attention_mask=None, causal_attention_mask=None)[0]
        
        # post-process and projection
        text_embedding = self.text_encoder.final_layer_norm(text_embedding)
        vision_embedding = self.vision_encoder.post_layernorm(vision_embedding)
        text_embedding = self.text_projection(text_embedding)   # [128, 77, 512]
        vision_embedding = self.vision_projection(vision_embedding)   # [128, 98, 512]
        
        # compute the final similarity
        # for the first similarity
        whole_text_representation = text_embedding[torch.arange(bsz), input_id.argmax(dim=-1)] #[B,512]
        whole_vision_representation = vision_embedding[:, 0, :]
        
        whole_text_representation = whole_text_representation / whole_text_representation.norm(dim=-1, keepdim=True)  #[1024,512]
        whole_vision_representation = whole_vision_representation / whole_vision_representation.norm(dim=-1, keepdim=True) #[1024,512]
        
        first_similarity = torch.mm(whole_text_representation, whole_vision_representation.T)
         # here we can rescale to (0, 1) and then use gamma transormation to enlarge the gap between comfortable region  
        first_similarity = (first_similarity + 1) / 2
        # here add a loss to limit the first similarity
        first_similarity = torch.diag(first_similarity)
        
        
        # for the second similarity
        batch_phrase_text_embedding = []
        for i, batch_position in enumerate(phrase_position):
            phrase_text_embedding = []
            for position in batch_position:
                if position[0] == -1 or position[1] == -1:
                    pass
                else:
                    phrase_text_embedding.append(torch.mean(text_embedding[i, position[0]:position[1]], dim=0))
            batch_phrase_text_embedding.append(torch.stack(phrase_text_embedding))

        
        # [B, 3, 512], 代表了三张图
        aux_vision_embedding = torch.cat([torch.mean(vision_embedding[:, 50: 66, :], dim=1, keepdim=True),
                                          torch.mean(vision_embedding[:, 66: 82, :], dim=1, keepdim=True),
                                          torch.mean(vision_embedding[:, 82: , :], dim=1, keepdim=True)], dim=1)
        
     
        second_similairty = self._many_to_many_sim_compute_try2(batch_phrase_text_embedding, aux_vision_embedding)#batch_phrase_text_embedding:len=1024  batch_phrase_text_embedding[i].shape:[num_phrase,512]
        
        token_vision_embedding = vision_embedding[:, 1:50, :]
        token_text_embedding = text_embedding
        
        # second_similairty = self._many_to_many_sim_compute_try(phrase_text_embedding, aux_vision_embedding, token_text_embedding, token_vision_embedding, attention_mask=attention_mask.unsqueeze(1))
        
        #attention_mask.unsqueeze(-1):[1024,77,1]
        third_similarity = self._many_to_many_sim_compute(token_text_embedding, token_vision_embedding, attention_mask=attention_mask.unsqueeze(-1))
        
        
        final_similarity = (first_similarity + second_similairty + third_similarity) / 3
        return final_similarity, data_id, first_similarity, second_similairty, third_similarity
        
    
        
        
            
        
        
    
        
        
            
        
        
        