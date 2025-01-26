import torch
from torch import nn, optim
from .base_model import SentenceRE
from torch.nn.functional import gelu,relu
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput

class SoftmaxNN(SentenceRE):   # 封装到这个softmax类的内部
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder   # 定义的用于抽取特征的encoder
        self.num_class = num_class   # 整体的用于分类的类别
        self.linear = nn.Linear(self.sentence_encoder.hidden_size, num_class)   # 这一层其实是没有用到的
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)   # 这里是最后进行分类的一层
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id  # label和text的对应
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()  # 用于推理的情况
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score    # 得到的类别和预测的得分

    def forward(self,*args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args)  # (B, H)，[16, 1536]
        rep = self.drop(rep)   # 进行了一个dropout
        #logits = F.relu(self.linear(rep))
        logits = self.fc(rep) # (B, N)   # 进行了一个fc层
        #logits = F.dropout(logits, 0.2)
        #logits = F.softmax(logits, dim=1)
        return logits, rep
