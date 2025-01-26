# coding:utf-8
import torch
import numpy as np
import json
import opennre.encoder, opennre.model, opennre.framework
from opennre.framework.data_loader import SimDataset, SimLoader, SentenceRELoader
import sys
import os
import argparse
import random
import math
import logging
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
                    help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='',
                    help='Checkpoint name')
parser.add_argument('--only_test', action='store_true', default=False,
                    help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--train_file', default='', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
                    help='Relation to ID file')

# Hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=1e-5, type=float,
                    help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=20, type=int,
                    help='Max number of training epochs')
parser.add_argument('--sample_ratio', default=1.0, type=float,
                    help="only for low resource.")
args = parser.parse_args()

# basic setting for the random seed(随机种子一定要在最早的时候就进行设定)
seed_number = 13
random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
torch.cuda.manual_seed_all(seed_number)
torch.backends.cudnn.deterministic = True  # 使得每一次卷积的输入和输出都是固定的

# Some basic settings
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt_path = 'ckpt/{}.pth.tar'.format(args.ckpt)

# text，训练、验证、测试集中的文本文件
root_path = ''
args.train_file = os.path.join(root_path, 'data', 'txt/ours_train.txt')
args.val_file = os.path.join(root_path, 'data', 'txt/ours_val.txt')
args.test_file = os.path.join(root_path, 'data', 'txt/ours_test.txt')
# original image
args.pic_train_file = os.path.join(root_path, 'data',  'img_org/train')   # 原数据集中的训练、验证、测试集
args.pic_val_file = os.path.join(root_path, 'data',  'img_org/val')   
args.pic_test_file = os.path.join(root_path, 'data',  'img_org/test')
# target relations
args.rel2id_file = os.path.join(root_path, 'data', 'ours_rel2id.json')   # 各种关系的id编号，对应的分类编号
if not os.path.exists(args.test_file):
    logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
    args.test_file = args.val_file
args.metric = 'micro_f1'   # 选用的评价测试指标是micro_f1

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
id2rel = {v: k for k, v in rel2id.items()}

# basic setting for our framework
sim_text_path = os.path.join(root_path, 'data', 'txt/new_ours_train.txt')
sim_pic_path = args.pic_train_file
sim_pretrained_path = 'clip-vit-base-patch32'
sim_batchsize = 256

# dataset and dataloader
sim_loader = SimLoader(sim_text_path, sim_pic_path, sim_pretrained_path, sim_batchsize, subset_flag=False, subset_index=None)

# define the model
clip_adaptive_sim = opennre.model.ClipSimAdaptive(freezing=True)
clip_adaptive_sim = torch.nn.DataParallel(clip_adaptive_sim) 


# Transfer the model to gpu
if torch.cuda.is_available():
    clip_adaptive_sim.cuda()  

no_decay = ['bias', 'layernorm.bias', 'layernorm.weight', 'layer_norm1.bias', 'layer_norm1.weight', 'layer_norm2.bias', 'layer_norm2.weight']

param_groups = [
    {'params': [ param for name, param in clip_adaptive_sim.named_parameters() if all([no_decay_ele not in name for no_decay_ele in no_decay])],
     'weight_decay': 0.2,
     'lr': 5e-4},
    {'params': [ param for name, param in clip_adaptive_sim.named_parameters() if any([no_decay_ele in name for no_decay_ele in no_decay])],
     'weight_decay': 0.0,
     'lr': 5e-4}]

# define the setting for curcriculum learning, change the ratio for difference experimental settings
curcriculum_ratio = 0.80    #  the original is 0.75, 0.80 is better
curcriculum_epoch = int(args.max_epoch * curcriculum_ratio)  #20*0.75=15

#try sigmoid scheduler
def compute_num_samples_for_curcriculum_learning(current_epoch, max_curriculum_epoch, start_coe=0.35, k=0.5):
    if current_epoch <= max_curriculum_epoch:
        # 调整Sigmoid函数，使其在t=0时输出start_value，在t=T时输出1
        t_0 = max_curriculum_epoch / 2  # Sigmoid的中点
        # 计算Sigmoid的输出，确保它从start_value开始，并在t=T时接近1
        adjusted_start = 1 / (1 + math.exp(-k * (0 - t_0)))  # t=0时的Sigmoid值
        end_value = 1 / (1 + math.exp(-k * (max_curriculum_epoch - t_0)))  # t=T时的Sigmoid值
        scale_factor = (1 - start_coe) / (end_value - adjusted_start)
        # 计算当前epoch的Sigmoid值，并按比例缩放
        current_sigmoid = 1 / (1 + math.exp(-k * (current_epoch - t_0)))
        scaled_sigmoid = (current_sigmoid - adjusted_start) * scale_factor + start_coe
        # 计算当前应使用的样本数量
        current_num_samples = math.ceil(scaled_sigmoid * 12247)
    else:
        current_num_samples = None  # 超过课程学习期后，使用全部样本
    return current_num_samples
#############

trianing_num_samples_for_curcriculum_learning_epoch = [compute_num_samples_for_curcriculum_learning(i, curcriculum_epoch, start_coe=0.35) for i in range(curcriculum_epoch)]
trianing_num_samples_for_curcriculum_learning_epoch.extend([12247]*(args.max_epoch - curcriculum_epoch))
training_steps_for_curcriculum_learning_epoch = [math.ceil(training_num_samples / args.batch_size) for training_num_samples in trianing_num_samples_for_curcriculum_learning_epoch]
num_training_steps = sum(training_steps_for_curcriculum_learning_epoch)

# define the optimizer and learning rate scheduler
sim_optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
num_warmup_steps = training_steps_for_curcriculum_learning_epoch[0]
sim_scheduler = get_cosine_schedule_with_warmup(optimizer=sim_optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)   

# Define the model，这里框死了cuda的所有随机种子，但是random和np.random里面的没有框住

# Define the sentence encoder,定义句子的encoder
sentence_encoder = opennre.encoder.TMR_RE(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity)

model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)    # 定义了一个用于分类的softmax函数，实际上是把它封装到了softmax这个函数的内部

# Define the whole training framework,用的是opennre这套框架进行的训练，把前面定义的所有全部都封装到了一起，包括了图也包括了文
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    train_pic_path=args.pic_train_file,
    val_path=args.val_file,
    val_pic_path=args.pic_val_file,
    test_path=args.test_file,
    test_pic_path=args.pic_test_file,
    model=model,
    ckpt=ckpt_path,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    warmup_step=num_warmup_steps,   #  set the warmup steps same as the other model
    train_step= num_training_steps,
    train_step_for_epoch=training_steps_for_curcriculum_learning_epoch,
    opt='adamw',
    sample_ratio=args.sample_ratio
)


# Train the model
if not args.only_test:
    metric='micro_f1'
    best_metric = 0
    global_step = 0
    train_loader = None
    
    #debug
    all_epoch_subset_indices = []

    init_sort_sim = None
    train_loss = []
    for i in range(framework.max_epoch):
            # run our framework for each epoch
        with torch.no_grad():
            whole_sim, whole_id = [], []
            whole_sim_first = []
            whole_sim_second = []
            whole_sim_third = []
            for sim_batch in tqdm(sim_loader):
                img_id, data_id, input_id, attention_mask, phrase_position, img, aux_img = sim_batch
                if torch.cuda.is_available():
                    # push the used data to gpu
                    input_id = input_id.cuda()  # [bsz, 77]
                    attention_mask = attention_mask.cuda()  # [bsz, 77]
                    phrase_position = phrase_position.cuda()  # [bsz, 6, 2]
                    img = img.cuda()    # [bsz, 3, 224, 224]
                    aux_img = aux_img.cuda()   # [bsz*3, 3, 128, 128]
                    batch_sim, batch_id, first_similarity, second_similairty, third_similarity = clip_adaptive_sim(input_id, attention_mask, phrase_position, img, aux_img, data_id)
                    whole_sim.append(batch_sim) 
                    whole_id.append(torch.tensor(batch_id))  #[tensor([   0,    1,    2,  ..., 1021, 1022, 1023]), tensor([1024, 1025, 1026,  ..., 2045, 2046, 2047])]
                    #case study
                    whole_sim_first.append(first_similarity)
                    whole_sim_second.append(second_similairty)
                    whole_sim_third.append(third_similarity)
            sim, id = torch.cat(whole_sim), torch.cat(whole_id)  #[12247]
            sort_sim, sort_indics = torch.sort(sim, descending=True)

                #case study
            sort_id = id[sort_indics]
            subset_index = sort_id[:trianing_num_samples_for_curcriculum_learning_epoch[i]].tolist()  
            
        #debug
        all_epoch_subset_indices.append(subset_index)
        # free the memory for the above computation
        torch.cuda.empty_cache()
        # here, check it by yourself, noth that adapt the resample in SentenceRELoader
        train_sim_loader = SimLoader(sim_text_path, sim_pic_path, sim_pretrained_path, args.batch_size, subset_flag=True, subset_indices=subset_index)
        new_train_loader = SentenceRELoader(args.train_file, args.pic_train_file, rel2id, model.sentence_encoder.tokenize, args.batch_size, shuffle=True, subset_flag=True, subset_indices=subset_index)
        # add the clipsimAdaptive forward and loss computation in new_train_model

        global_step, best_metric, clip_adaptive_sim, sim_optimizer, sim_scheduler,train_loss = framework.new_train_model(train_loss, clip_adaptive_sim, sim_optimizer, sim_scheduler, epoch=i, global_step=global_step, sim_loader= train_sim_loader, train_loader=new_train_loader, best_metric=best_metric, split_ratio=0.5, metric=metric)


    logging.info("Best %s on val set: %f" % (metric, best_metric))


# Val
for loader in [framework.val_loader]:
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Val set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))

# Test
for loader in [framework.test_loader]:
    framework.load_state_dict(torch.load(ckpt_path)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Test set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))
    with open('./results/test/' + args.ckpt + '_result3123.txt', 'w') as f:
        for i in range(len(rel2id)):
            f.write(str(id2rel[i]) + ':' + str(acc_category[i]) + 'the count is ' + str(n_category[i]) + '\n')
        f.write('Test set results: \n')
        f.write('Accuracy: {}\n'.format(result['acc']))
        f.write('Micro precision: {}\n'.format(result['micro_p']))
        f.write('Micro recall: {}\n'.format(result['micro_r']))
        f.write('Micro F1: {}\n'.format(result['micro_f1']))
    with open('./results/test/' + args.ckpt + '_data_with_pred_t123124.json', 'w', encoding='UTF-8') as f1:
        for i in range(len(data_pred_t)):
            f1.write(data_pred_t[i] + "\n")
    with open('./results/test/' + args.ckpt + '_data_with_pred_f412312.json', 'w', encoding='UTF-8') as f2:
        for i in range(len(data_pred_f)):
            f2.write(data_pred_f[i] + "\n")
