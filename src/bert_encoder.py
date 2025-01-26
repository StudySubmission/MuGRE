import logging
import torch
import torch.nn as nn
from transformers import BertTokenizer
import transformers
import math
from torch.nn import functional as F
import json
from torch.nn.functional import gelu, relu, tanh
import numpy as np
from torch import nn
from torchvision.models import resnet50
import timm


class TMR_RE(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        max_length: max length of sentence
        pretrain_path: path of pretrain model
        blank_padding: need padding or not
        mask_entity: mask the entity tokens or not
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding   # 这里设置的是文本的句子是需要padding的
        self.mask_entity = mask_entity       # 是否需要对实体的token进行mask
        self.hidden_size = 768*2             # 隐藏层的大小

        self.nclass = 300    # 图像用的是resnet50系列，但是这个nclass有点奇怪
        self.model_resnet50 = timm.create_model('./resnet50') # get the pre-trained ResNet model for the image
        for param in self.model_resnet50.parameters():
            param.requires_grad = True
        # 文本这边用的是bert的编码器
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained(pretrain_path) # get the pre-trained BERT model for the text
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_pic = nn.Linear(2048, self.hidden_size//2)    # 给图像resnet50编码得到的2048维向量降维
        self.linear_final = nn.Linear(self.hidden_size*2 + self.hidden_size, self.hidden_size)   # 用于最后进行分类的一层

        # the attention mechanism for fine-grained features,细粒度特征的attention
        self.linear_q_fine = nn.Linear(768, self.hidden_size//2)   # 除以2其实就是768
        self.linear_k_fine = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.linear_v_fine = nn.Linear(self.hidden_size//2, self.hidden_size//2)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.linear_k_coarse = nn.Linear(self.hidden_size//2, self.hidden_size//2)
        self.linear_v_coarse = nn.Linear(self.hidden_size//2, self.hidden_size//2)

        self.linear_weights = nn.Linear(self.hidden_size*3, 3)   # 这一层不知道干什么的
        self.linear_phrases = nn.Linear(self.hidden_size//2, self.hidden_size)  # 短语特征这一块进行加权
        self.linear_extend_pic = nn.Linear(self.hidden_size//2, self.hidden_size//2)  # 对变化后的两层进行求和得到相应的结果
        self.dropout_linear = nn.Dropout(0.5)   # 进行一层dropout

    def forward(self, token, att_mask, pos1, pos2, token_phrase, att_mask_phrase, image_diff, image_ori, image_diff_objects, image_ori_objects, weights):

        # 这里才是整个函数的主体部分，输入的部分包括了输入的处理好的token和对应的attention mask,头实体的位置，尾实体的位置，
        feature_DifImg_FineGrained = self.model_resnet50.forward_features(image_diff) # [16, 2048, 7, 7], resnet编码的结果
        feature_OriImg_FineGrained = self.model_resnet50.forward_features(image_ori)  # [16, 2048, 7, 7], resnet编码的结果
        feature_DifImg_CoarseGrained = self.model_resnet50.forward_features(image_diff_objects)  # [48, 3, 224, 224]  # 每一个最多是3，batchsize是16，所以是48
        feature_OriImg_CoarseGrained = self.model_resnet50.forward_features(image_ori_objects)  # [48, 3, 224, 224]  # 每一个最多是3，batchsize是16，所以是48

        pic_diff = torch.reshape(feature_DifImg_FineGrained, (-1,2048,49))  # [16, 2048, 49]
        pic_diff = torch.transpose(pic_diff, 1, 2)   # [16, 49, 2048]
        pic_diff = torch.reshape(pic_diff, (-1, 49, 2048))  # [16, 49, 2048]
        pic_diff = self.linear_pic(pic_diff)  # 降维成[16, 49, 768]
        pic_diff_ = torch.sum(pic_diff, dim=1)  # 整体取了一个平均[16, 768]
        # 和上面是一样的操作，只不过处理的是original的图像
        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)
        pic_ori_ = torch.sum(pic_ori,dim=1) # 整体取了一个平均[16, 768]
        # 处理diffusion model得到的visual grounding的结果
        pic_diff_objects = torch.reshape(feature_DifImg_CoarseGrained, (-1, 2048, 49))  # [48, 2048, 49]
        pic_diff_objects = torch.transpose(pic_diff_objects, 1, 2)  # # [48, 49, 2048]
        pic_diff_objects = torch.reshape(pic_diff_objects, (-1, 3, 49, 2048))    #pic_diff_objects = torch.reshape(pic_diff_objects, (16, 3, 49, 2048)) 
        pic_diff_objects = torch.sum(pic_diff_objects, dim=2)   # [16, 3, 2048]，在第二个维度上取平均
        pic_diff_objects = self.linear_pic(pic_diff_objects)   # 和上面依然是公用的 [16, 3, 768]
        pic_diff_objects_ = torch.sum(pic_diff_objects,dim=1)   # 再在第一个维度上取平均[16, 768]
        # 处理original得到的visual grounding的结果，和上面是一样的操作，只不过换一个输入
        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects = self.linear_pic(pic_ori_objects)   # [16, 3, 768]
        pic_ori_objects_ = torch.sum(pic_ori_objects,dim=1)  # 再在第一个维度上取平均[16, 768]
        # '[CLS] sent0 [SEP]', 100, 101, padding 0， 1 1 1 1 1 0 0 0 0
        output_text = self.bert(token, attention_mask=att_mask)  # 用bert来处理文本
        hidden_text = output_text[0]   # [16, 128, 768]，返回最后一层的每一个token
        # ??? att_mask_phrase 0 0 0 0 0
        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)  # 用bert编码文本，这个时候其实全部都是0
        hidden_phrases = output_phrases[0]
        # 是用文本来增强图像，???
        hidden_k_text = self.linear_k_fine(hidden_text)   # k的变换
        hidden_v_text = self.linear_v_fine(hidden_text)   # v的变换
        pic_q_diff = self.linear_q_fine(pic_diff)         # q的变换
        pic_diffusion = torch.sum(torch.tanh(self.att(pic_q_diff, hidden_k_text, hidden_v_text)), dim=1)  # 做了一个平均(16, 768)
        # 同样是用文本来增强图像
        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)  # 做了一个平均(16, 768)
        # 依然是用文本来增强图像
        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_diff_objects = self.linear_q_coarse(pic_diff_objects)  # 这里是三个图像
        pic_diffusion_objects = torch.sum(torch.tanh(self.att(pic_q_diff_objects, hidden_k_phrases, hidden_v_phrases)), dim=1)  # 做了一个平均[16, 768]
        # 依然是用文本来增强图像
        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)), dim=1)

        # coarse-grained textual features
        hidden_phrases = torch.sum(hidden_phrases, dim=1)  # [16, 768]
        hidden_phrases = self.linear_phrases(hidden_phrases)

        # Get entity start hidden state
        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)  # 把对应的pos1的位置设置为1
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)  # 把对应的pos2的位置设置为1
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)  # (B, H)   # 抽取出头实体的矢量[16, 768]
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)  # (B, H)   # 抽取出尾实体的矢量
        # fine-grained textual features
        x = torch.cat([head_hidden, tail_hidden], dim=-1)   # 得到的就是[16, 1536]

        pic_ori_final = (pic_original+pic_ori_) * weights[:, 1].reshape(-1, 1) + (pic_original_objects+pic_ori_objects_) * weights[:, 0].reshape(-1,1)
        pic_diff_final = (pic_diffusion+pic_diff_) * weights[:, 3].reshape(-1, 1) + (pic_diffusion_objects+pic_diff_objects_) * weights[:, 2].reshape(-1, 1)
        
        pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final))   # 得到的是[16, 768]
        pic_diff = torch.tanh(self.linear_extend_pic(pic_diff_final))  # 得到的是[16, 768], 且它们是共用的
        
    
        x = torch.cat([x, hidden_phrases, pic_ori, pic_diff], dim=-1)
        x = self.linear_final(self.dropout_linear(x))

        return x

    def tokenize(self, item):     # 传进来的是编写分词过程的这个函数，封装在TMR_RE的这个模型当中,item这一项目包括了几个东西
 
        if len(item) == 5 and not isinstance(item, dict):
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(item)
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

            return indexed_tokens

        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        elif 'token' in item:
            sentence = item['token']
            is_token = True   # 已经转化成了token

        pos_head = item['h']['pos']   # 得到了头实体的位置
        pos_tail = item['t']['pos']   # 得到了尾实体的位置

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:    # 头实体的位置在尾实体之后，那么就调转过来
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        # 前面已经分词完毕了，这里要连成句子一起操作
        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])    # 它标记的一定是处于前面的那个实体
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])    # 它标记的一定是处于后面的那个实体
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:  # 重新用bert的tokenizer来进行一遍分词
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))    # entity0之前的所有词汇构成的句子
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))  # entity0
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))  # entity0和entity1之间的句子
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))  # entity1
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))  # entity1之后的句子

        if self.mask_entity:    # 是否要对entity进行mask
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:     # 这里也加了一些序列的标号来标记两个实体的位置，rev=False的时候，头实体在前面，ent0=['[unused0]'] + ent0 + ['[unused1]']，尾实体在后面，rev=True的时候，则头实体在后面，这里用unused的标号来标记了头尾实体的先后顺序，也就是2和3之间一定是尾实体，0和1之间一定是头实体
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']
        # 这里人工的自己加上了['CLS']的标签，用于bert，且最后加了'[SEP]'这个分割符
        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)   # pos1指的是头实体的位置，前面有一个['CLS']抵消，所以这里加上1
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)   # pos2指的是尾实体的位置，和上面是一样的
        pos1 = min(self.max_length - 1, pos1)   # 避免超出了tokenizer的位置限制
        pos2 = min(self.max_length - 1, pos2)   # 这里也是避免超出了tokenizer的限制
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)   # 把输入的整体转化成token,其中CLS的token是101，sep的token是102
        avai_len = len(indexed_tokens)   # 代表它的输入长度

        # Position
        pos1 = torch.tensor([[pos1]]).long()   # 指定两者的不同位置
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:   # 需要进行padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1   # 用于attention的部分设置为1，不用于attention的部分是设置为0

        phrases = item['grounding']   # 把短语的这个部分转化成token_id
        token_phrases =self.tokenizer.convert_tokens_to_ids(phrases.split(' '))
        while len(token_phrases)<6:   # token_phrases的长度最大是6
            token_phrases.append(0)
        token_phrases = token_phrases[:6]  # 只取前面的6位
        token_phrases = torch.tensor(token_phrases).long().unsqueeze(0)
        att_mask_phrases = torch.zeros(token_phrases.size()).long()

        return indexed_tokens, att_mask, pos1, pos2, token_phrases, att_mask_phrases  

    # the attention mechanism
    def att(self, query, key, value):   # 进行attention运算的部分
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)   # q,k先算
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)  # 然后和value又乘了一遍

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                # x = gelu(x)
                x = relu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x




