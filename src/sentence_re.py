import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
import math
from .utils import AverageMeter
import numpy as np
from sklearn import metrics
from seqeval.metrics import f1_score

class SentenceRE(nn.Module):   
    def __init__(self,
                 model,
                 train_path,
                 train_pic_path,
                 val_path,
                 val_pic_path,
                 test_path,
                 test_pic_path,
                 ckpt,
                 batch_size=64,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-3,
                 warmup_step=376,
                 train_step=None,
                 train_step_for_epoch=None,
                 opt='adamw',
                 sample_ratio=1.0):

        super().__init__()
        self.max_epoch = max_epoch    # 最大的epoch
        # Load data
        pretrained_path = '/root/autodl-tmp/workspace/TMP_ACL2023/TMR/TMR-RE/clip-vit-base-patch32'
        sim_batchsize = 128
        if train_path != None:
            # fr
            self.train_loader = SentenceRELoader(   # 对train的dataloader进行的设置
                train_path,
                train_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,   # 这里是把tokenize的函数放了进去进行操作
                batch_size,
                True,
                sample_ratio=sample_ratio)
            
        # 这两个部分都是不需要的
        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                val_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                sample_ratio=sample_ratio)
        # 这个部分也是不需要的
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                test_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                sample_ratio=sample_ratio)
        # Model
        self.model = model   #  这里是原始的model，在这里应该插入用于课程学习的模型
        self.parallel_model = nn.DataParallel(self.model)    # 给它开多机，多卡设置了并行操作
        # Criterion
        self.train_criterion = nn.CrossEntropyLoss(reduction='none')   # 设置计算loss的方式
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()   # 定义的模型中所有的参数
        self.lr = lr
        if opt == 'sgd':   # 优化器的设定
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']   # 设置no_decay的步骤
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        
        if warmup_step > 0:   # warmup step的设定
            from transformers import get_linear_schedule_with_warmup
            #training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch  #766
            training_steps = train_step
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        

        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0   # 当下的best metric是多少
        global_step = 0   # 全局的step是多少
        loader = [self.train_loader]   # 封装好的数据集加载器
        for epoch in range(self.max_epoch):
            self.train()    # 设置当前模型的状态为训练模式
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()   # 这些都是用来记录的模块
            avg_acc = AverageMeter()
            avg_f1 = AverageMeter()
            for l1 in loader:   # 读取每一个loader
                t = tqdm(l1, ncols=66)   # 从这个loader里面读取每一个batch,ncols是用来指定进度条的宽度的
                for iter, data in enumerate(t):   # 读取每一个batch，得到每一项的data
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()   # 这里逐项数据搬运也是很有意思的一个部分
                            except:
                                pass
                    # data中的每一项分别是：样本对应的label,样本对应的img_id，tokenize处理好后的seq,原图和扩散图的整图，原图和扩散图的vg图，最后的权重分数
                    label = data[0]
                    img_id = data[1]
                    args = data[2:]
                    logits, rep = self.parallel_model(*args)  # 前面返回的是logits值，没什么用的[16, 23]
                    loss = self.criterion(logits, label).mean()   # 用Crossentropyloss
                    score, pred = logits.max(-1)  # (B)    # 取预测类别
                    acc = float((pred == label).long().sum()) / label.size(0)   # 算准确率
                    f1 = metrics.f1_score(pred.cpu(), label.cpu(), average='macro')   # 算f1的📄
                    # Log
                    avg_loss.update(loss.item(), 1)   # 记录每一项更新的值
                    avg_acc.update(acc, 1)
                    avg_f1.update(f1, 1)
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, f1=avg_f1.avg)  # 这里用进度条显示了每一个batch里预测的值的正确与否
                    # Optimize
                    loss.backward()   # 梯度回传
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()   # 梯度scheduler的更新
                    self.optimizer.zero_grad()  # 清空梯度
                    global_step += 1

            # Val，每一个epoch进行一次测评
            logging.info("=== Epoch %d val ===" % epoch)
            result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = self.eval_model(
                self.val_loader)
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                best_metric = result[metric]   # 记录最好的标签
                # 这里是用来存储的一个部分
                folder_path = '/'.join(self.ckpt.split('/')[:-1])   # 存ckpt
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                logging.info("Best ckpt and saved.")
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def new_ablation_train_model(self, epoch, global_step, train_loader, best_metric, split_ratio=0.6, metric='acc'):
        global_step = global_step   # 全局的step是多少
        self.train()    # 设置当前模型的状态为训练模式
        logging.info("=== Epoch %d train ===" % epoch)
        avg_re_loss = AverageMeter()   # 这些都是用来记录的模块
        avg_acc = AverageMeter()
        avg_f1 = AverageMeter()
        avg_sim_loss = AverageMeter()
        avg_sim_norm_loss = AverageMeter()
        avg_total_loss = AverageMeter()
        avg_first_sim = AverageMeter()
        avg_second_sim = AverageMeter()
        avg_third_sim = AverageMeter()
        avg_final_sim = AverageMeter()  
        # assert len(sim_loader.dataset) == len(train_loader.dataset)
        with tqdm(total=len(train_loader), ncols=110) as t:
            
            for re_batch in tqdm(train_loader):   # 读取每一个loader
                # sim_img_id, sim_data_id, sim_input_id, sim_attention_mask, sim_phrase_position, sim_img, sim_aux_img = sim_batch
                if torch.cuda.is_available():
                    # sim_input_id = sim_input_id.cuda()
                    # sim_attention_mask = sim_attention_mask.cuda()
                    # sim_phrase_position = sim_phrase_position.cuda()
                    # sim_img = sim_img.cuda()
                    # sim_aux_img = sim_aux_img.cuda()
                    for i in range(len(re_batch)):
                        try:
                            re_batch[i] = re_batch[i].cuda()
                        except:
                            pass
                    
                    re_label = re_batch[0]
                    re_img_id = re_batch[1]
                    re_args = re_batch[2:]
                
                # forward the relation extraction
                re_logits, re_rep = self.parallel_model(*re_args)
                re_loss = self.train_criterion(re_logits, re_label)   #关系抽取loss[B]        
                score, pred = re_logits.max(-1)  # (B)    # 取预测类别
                acc = float((pred == re_label).long().sum()) / re_label.size(0)   # 算准确率
                f1 = metrics.f1_score(pred.cpu(), re_label.cpu(), average='macro')   # 算f1的📄
                
                #  forward the similarity 
               

                # compute the total batch loss
                total_loss = re_loss.mean() 
                
                
                """
                # mask with auto-adjusted threshold
                
                threshold = re_loss.mean()    # compute the mean value
                mask = torch.zeros_like(batch_sim, dtype=torch.bool)   
                mask = re_loss > threshold   # compute the mask matrix for decoding which to negtive
                mask_index = torch.nonzero(mask).squeeze()
                negative_re_loss = re_loss[mask]
                positive_re_loss = torch.exp(-0.5 * re_loss[~mask])
                scale_negative_re_loss = negative_re_loss / negative_re_loss.mean()
                scale_positive_re_loss = positive_re_loss / positive_re_loss.mean()
                scale_negative_re_loss_factor =  scale_negative_re_loss.clone().detach()
                scale_positive_re_loss_factor = scale_positive_re_loss.clone().detach()
                scale_negative_re_loss_factor.requires_grad = False
                scale_positive_re_loss_factor.requires_grad = False
                
                sim_loss_norm = torch.zeros_like(batch_sim)
                pos_index, neg_index = 0, 0
                for i in range(len(sim_loss_norm)):
                    if i in mask_index:
                        sim_loss_norm[i] = - scale_negative_re_loss[neg_index] * torch.log( 1 - batch_sim[i])
                        neg_index = neg_index + 1
                    else:
                        sim_loss_norm[i] = - scale_positive_re_loss[pos_index] * torch.log(batch_sim[i])
                        pos_index = pos_index + 1

                total_loss = re_loss.mean() + sim_loss_norm.mean()
                """
                
                # Log

                avg_total_loss.update(total_loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_f1.update(f1, 1)

                t.set_postfix(
                    total_loss=avg_total_loss.avg, 
                    acc=avg_acc.avg, 
                    f1=avg_f1.avg,
                )

                
                # Optimize
                total_loss.backward()   # 梯度回传
                self.optimizer.step()
                # sim_optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()   # 梯度scheduler的更新
                # if sim_scheduler is not None:
                #     sim_scheduler.step()
                self.optimizer.zero_grad()  # 清空梯度
                # sim_optimizer.zero_grad()
                global_step += 1

        # Val，每一个epoch进行一次测评
        logging.info("=== Epoch %d val ===" % epoch)
        result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = self.eval_model(
            self.val_loader)
        logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
        if result[metric] > best_metric:
            best_metric = result[metric]   # 记录最好的标签
            # 这里是用来存储的一个部分
            folder_path = '/'.join(self.ckpt.split('/')[:-1])   # 存ckpt
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
            logging.info("Best ckpt and saved.")
        return global_step, best_metric

    def new_train_model(self,train_loss, clip_adaptive_sim, sim_optimizer, sim_scheduler, epoch, global_step, sim_loader, train_loader, best_metric, split_ratio=0.5, metric='acc'):
        global_step = global_step   # 全局的step是多少
        self.train()    # 设置当前模型的状态为训练模式
        logging.info("=== Epoch %d train ===" % epoch)
        avg_re_loss = AverageMeter()   # 这些都是用来记录的模块
        avg_acc = AverageMeter()
        avg_f1 = AverageMeter()
        avg_sim_loss = AverageMeter()
        avg_sim_norm_loss = AverageMeter()
        avg_total_loss = AverageMeter()
        avg_first_sim = AverageMeter()
        avg_second_sim = AverageMeter()
        avg_third_sim = AverageMeter()
        avg_final_sim = AverageMeter()  
        assert len(sim_loader.dataset) == len(train_loader.dataset)
        train_batch_loss=0
        with tqdm(total=len(sim_loader), ncols=110) as t:
            
            for sim_batch, re_batch in tqdm(zip(sim_loader, train_loader)):   # 读取每一个loader
                sim_img_id, sim_data_id, sim_input_id, sim_attention_mask, sim_phrase_position, sim_img, sim_aux_img = sim_batch
                if torch.cuda.is_available():
                    sim_input_id = sim_input_id.cuda()
                    sim_attention_mask = sim_attention_mask.cuda()
                    sim_phrase_position = sim_phrase_position.cuda()
                    sim_img = sim_img.cuda()
                    sim_aux_img = sim_aux_img.cuda()
                    for i in range(len(re_batch)):
                        try:
                            re_batch[i] = re_batch[i].cuda()
                        except:
                            pass
                    
                    re_label = re_batch[0]
                    re_img_id = re_batch[1]
                    re_args = re_batch[2:]
                
                # forward the relation extraction
                re_logits, re_rep = self.parallel_model(*re_args)
                re_loss = self.train_criterion(re_logits, re_label)   #关系抽取loss[B]        
                score, pred = re_logits.max(-1)  # (B)    # 取预测类别
                acc = float((pred == re_label).long().sum()) / re_label.size(0)   # 算准确率
                f1 = metrics.f1_score(pred.cpu(), re_label.cpu(), average='macro')   # 算f1的📄
                
                #  forward the similarity 
                batch_sim, batch_id, first_similarity, second_similairty, third_similarity = clip_adaptive_sim(sim_input_id, sim_attention_mask, sim_phrase_position, sim_img, sim_aux_img, sim_data_id)#[B]
                
                
                ###正常   ###正常  ###正常
                # negative sample masking
                # mask with ratio
                
                mask_num = math.ceil(len(batch_sim) * split_ratio)
                mask_indics = torch.sort(re_loss, descending=True)[1][:mask_num]
                mask = torch.zeros_like(batch_sim, dtype=torch.bool)
                mask[mask_indics] = True
                
                # compute the loss for sim loss
                sim_loss = torch.zeros_like(batch_sim)
                for i in range(len(sim_loss)):
                    if mask[i] == 1:
                        sim_loss[i] = - torch.log(1 - batch_sim[i])
                    else:
                        sim_loss[i] = - torch.log(batch_sim[i])
                # compute the mean for positive and negative samples
                
            
                negative_mean = re_loss[mask].mean()
            
                ##negative try ->torch.exp(0.5x)"
                # negative_mean = torch.exp(0.6*re_loss[mask]).mean()
                # let the smaller number occupy the large proportion
                positive_mean = torch.exp(-0.75*re_loss[~mask]).mean()
                normalized_scale_factor = torch.zeros_like(batch_sim)
                
                normalized_scale_factor[mask] =  re_loss[mask] / negative_mean
            
                # normalized_scale_factor[mask] =  torch.exp(0.6*re_loss[mask]) / negative_mean
                normalized_scale_factor[~mask] = torch.exp(-0.75*re_loss[~mask]) / positive_mean
                
                scale_matrix = normalized_scale_factor.clone().detach()
                scale_matrix.requires_grad = False
                
                # scale the loss 
                sim_loss_norm = sim_loss * scale_matrix 
                ###正常  ###正常  ###正常
                

                """
                ##ablation_feedback
                sim_loss_norm = - torch.log(batch_sim)
                ##ablation_feedback
                """

                # compute the total batch loss
                total_loss = re_loss.mean() + sim_loss_norm.mean()
                train_batch_loss += re_loss.mean().item()
                
                """
                # mask with auto-adjusted threshold
                
                threshold = re_loss.mean()    # compute the mean value
                mask = torch.zeros_like(batch_sim, dtype=torch.bool)   
                mask = re_loss > threshold   # compute the mask matrix for decoding which to negtive
                mask_index = torch.nonzero(mask).squeeze()
                negative_re_loss = re_loss[mask]
                positive_re_loss = torch.exp(-0.5 * re_loss[~mask])
                scale_negative_re_loss = negative_re_loss / negative_re_loss.mean()
                scale_positive_re_loss = positive_re_loss / positive_re_loss.mean()
                scale_negative_re_loss_factor =  scale_negative_re_loss.clone().detach()
                scale_positive_re_loss_factor = scale_positive_re_loss.clone().detach()
                scale_negative_re_loss_factor.requires_grad = False
                scale_positive_re_loss_factor.requires_grad = False
                
                sim_loss_norm = torch.zeros_like(batch_sim)
                pos_index, neg_index = 0, 0
                for i in range(len(sim_loss_norm)):
                    if i in mask_index:
                        sim_loss_norm[i] = - scale_negative_re_loss[neg_index] * torch.log( 1 - batch_sim[i])
                        neg_index = neg_index + 1
                    else:
                        sim_loss_norm[i] = - scale_positive_re_loss[pos_index] * torch.log(batch_sim[i])
                        pos_index = pos_index + 1

                total_loss = re_loss.mean() + sim_loss_norm.mean()
                """
                
                # Log
                avg_re_loss.update(re_loss.mean().item(), 1)   # 记录每一项更新的值
                avg_sim_loss.update(sim_loss_norm.mean().item(), 1)
                avg_sim_norm_loss.update(sim_loss_norm.mean().item(), 1)
                avg_total_loss.update(total_loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_f1.update(f1, 1)
                avg_first_sim.update(first_similarity.mean().item(), 1)
                avg_second_sim.update(second_similairty.mean().item(), 1)
                avg_third_sim.update(third_similarity.mean().item(), 1)
                avg_final_sim.update(batch_sim.mean().item(), 1)
                t.set_postfix(
                    re_loss=avg_re_loss.avg, 
                    sim_loss=avg_sim_loss.avg,
                    total_loss=avg_total_loss.avg, 
                    acc=avg_acc.avg, 
                    f1=avg_f1.avg,
                )
                logging.info('first_sim:{} second_sim:{} third_sim:{} final_sim:{}'.format(avg_first_sim.avg, avg_second_sim.avg, avg_third_sim.avg, avg_final_sim.avg)
                             )
                
                # Optimize
                total_loss.backward()   # 梯度回传
                self.optimizer.step()
                sim_optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()   # 梯度scheduler的更新
                if sim_scheduler is not None:
                    sim_scheduler.step()
                self.optimizer.zero_grad()  # 清空梯度
                sim_optimizer.zero_grad()
                global_step += 1
            train_batch_loss /= 16
            train_loss.append(train_batch_loss)

        # Val，每一个epoch进行一次测评
        logging.info("=== Epoch %d val ===" % epoch)
        result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = self.eval_model(
            self.val_loader)
        logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
        if result[metric] > best_metric:
            best_metric = result[metric]   # 记录最好的标签
            # 这里是用来存储的一个部分
            folder_path = '/'.join(self.ckpt.split('/')[:-1])   # 存ckpt
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
            logging.info("Best ckpt and saved.")
        return global_step, best_metric, clip_adaptive_sim, sim_optimizer, sim_scheduler,train_loss


    def eval_model(self, eval_loader):
        self.eval()   # 设置到测评模式
        avg_acc = AverageMeter()   # 记录每一项操作的设置
        avg_loss = AverageMeter()
        pred_result = []   # 存储不同值的list
        id_list = []
        feature_list = []
        with torch.no_grad():
            t = tqdm(eval_loader, ncols=110)  # 同样是设置进度条
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                img_id = data[1]
                args = data[2:]
                logits, rep = self.parallel_model(*args)  # 一样的操作
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1)  # (B)
                id_list.append(img_id)   # 测评的每一个图像id的记录
                feature_list.append(rep)   # 得到的最后一层的feature的记录
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())   # 预测结果的记录

                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                avg_loss.update(loss.item(), pred.size(0))
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
        result, correct_category, org_category, n_category, data_pred_t, data_pred_f = eval_loader.dataset.eval(
            pred_result)
        # save prediction into JSON，把预测的结果存到json中
        with open('./pred_results.josn', 'w') as f2:
            json.dump(pred_result, f2)
        return result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


