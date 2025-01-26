import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
import timm
import cv2
from tqdm import tqdm, trange
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
from transformers import CLIPProcessor

class SimDataset(data.Dataset):


    def __init__(self, text_path, pic_path, pretrained_path, kwargs):

        
        super().__init__()
        # 文本路径
        self.text_path = text_path
        # clip中的数据处理块
        self.processor = CLIPProcessor.from_pretrained(pretrained_path)
        self.aux_processor = CLIPProcessor.from_pretrained(pretrained_path)
        self.aux_processor.feature_extractor.size, self.aux_processor.feature_extractor.crop_size = 128, 128
        mode = 'train'  
        # 原图的路径
        self.pic_path_FineGrained_ori = pic_path    
        # 原图visual grounding的路径
        self.pic_path_CoarseGrained_ori = pic_path.replace('org', 'vg')    

        # 加载文本的数据集
        self.data = []   
        f = open(text_path, encoding='UTF-8')  
        f_lines = f.readlines()
        for i1 in tqdm(range(len(f_lines))):
            # 去掉读入txt文件时写的换行符
            line = f_lines[i1].rstrip()   
            if len(line) > 0:
                # txt转字典
                dic1 = eval(line)   
                # 统一存储
                self.data.append(dic1)   
        f.close()
        # 分类类别转化成数字id 
        logging.info(
            "Loaded Similarity RE dataset {} with {} lines.".format(text_path, len(self.data)))

        # 获得aux图像和原图一一对应的字典路径
        self.img_aux_path_ori = text_path.replace('ours_{}.txt'.format(mode), 'mre_{}_dict.pth'.format(mode)).replace('new_','')
   
        # 下面的这个对应是原图中rcnn得到的一系列路径，但是它的键其实是它的数字id标号，这一点是要值得注意的
        self.state_dict_ori = torch.load(self.img_aux_path_ori)
           
        # 获得每一条样本中抽取出的短语，它的键值是数字标号，这个标号也是提前和self.data是对齐的
        self.phrase_path = text_path.replace('ours_{}.txt'.format(mode), 'phrase_text_{}.json'.format(mode)).replace('new_', '')
        self.phrase_data = json.load(open(self.phrase_path, 'r')) 

    # 实际上dataset中一共得到了打乱后的self.data, self.state_dict_dif, self.state_dict_ori, self.phrase_data这几样，他们都是对齐的，而strong和weak的分数则是根据img_id进行读取的
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):    # 开始依次取出相应的样本，这里直接顺序读取即可，没有必要采用shuffle
        # 读取某一个图像的image_id   
        # 存储original images，下面用于读取扩散模型生成的图像
        aux_imgs_ori = []     

        for i in range(3):   # rcnn中检测出来的object
            # rcnn中检测出来的object和原图的对应关系
            if len(self.state_dict_ori[index])>i:    
                image_ori_object = Image.open(os.path.join(self.pic_path_CoarseGrained_ori, 'crops/' + self.state_dict_ori[index][i])).convert('RGB')
                img_features_ori = self.aux_processor(images=image_ori_object, return_tensors='np')['pixel_values'].squeeze().tolist() # 存放的是一个list[3, 128, 128]
                aux_img_ori = img_features_ori
                # 处理完之后每一个图像得到的是一个列表，代表RGB三个通道
                aux_imgs_ori.append(aux_img_ori)   

        # 不足的全部padding成0
        for i in range(3 - len(aux_imgs_ori)):
            aux_imgs_ori.append(torch.zeros((3, 128, 128)).tolist())    

        assert len(aux_imgs_ori) == 3
        # 读取这条样本的文本数据,item中含有的键包括了'token','h','t','img_id','relation','phrase_position','id'
        item = self.data[index]    
        # 读取这个样本对应的短语，然后切分成列表
        item['grounding'] = self.phrase_data[str(index)].split()
        # 读取一下图像的id便于对照
        img_id = item['img_id']
        # 这里是需要取出样本的id的
        data_id = item['id']
        
        # 这个部分开始对文本进行分词，确保能够找到phrase的position已经截断过长的样本
        sentence = item['token']
        
        # 确认首尾实体的位置
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']
        pos_min = pos_head
        pos_max = pos_tail
        # 判断头尾实体的位置是否相反，必须是要大于才可以，公共重叠的不算
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        # 如果出现了反转，那说明尾实体在前，那么一开始加的标记应该是给尾实体加的，并且一定说明尾实体的左侧要更小一些
        if rev:
            sentence.insert(pos_min[0], '[unused2]') 
            # 加完标记以后，由于它是最小的，因此后续的全部索引都加上1
            pos_min[1] = pos_min[1] + 1
            pos_max[0] = pos_max[0] + 1
            pos_max[1] = pos_max[1] + 1
            # 给尾实体的另一侧加上标记
            sentence.insert(pos_min[1], '[unused3]')
            # 分情况讨论，如果尾实体的最右侧是小于或者等于头实体的最左侧的，那么两者完全分开,done
            if pos_min[1] <= pos_max[0]:
                # 完全分开的情况下索引继续加1
                pos_max[0] = pos_max[0] + 1
                pos_max[1] = pos_max[1] + 1
                # 插入头实体左侧的标记
                sentence.insert(pos_max[0], '[unused0]')
                # 右侧索引继续加1
                pos_max[1] = pos_max[1] + 1
                # 插入头实体右侧的标记
                sentence.insert(pos_max[1], '[unused1]')
            else:
                # 如果尾实体的最右侧是大于头实体的最左侧的，那么两者出现重叠的部分
                # 分情况讨论，如果尾实体完全将头实体包裹,则尾实体索引不需要动, done
                if pos_min[1] >= pos_max[1]:  
                    sentence.insert(pos_max[0], '[unused0]')  
                    # 它的索引加1
                    pos_max[1] = pos_max[1] + 1
                    # 按照先来后到的原则，再对应插入即可，此时它会出现在它的左侧
                    sentence.insert(pos_max[1], '[unused1]')
                else:
                    # 则尾实体没有完全将头实体包裹，说明头实体的左侧是在尾实体里面，头实体右侧在尾实体外面
                    pos_max[1] = pos_max[1] + 1
                    sentence.insert(pos_max[0], '[unused0]')
                    # 插入之后继续加1
                    pos_max[1] = pos_max[1] + 1
                    sentence.insert(pos_max[1], '[unused1]')
        else:
            # 这种情况是没有出现反转的，那么是头实体在前，尾实体在后，只是头实体和尾实体的左侧有可能重叠
            sentence.insert(pos_min[0], '[unused0]')
            # 这里默认头实体是最小的
            pos_min[1] = pos_min[1] + 1
            pos_max[0] = pos_max[0] + 1
            pos_max[1] = pos_max[1] + 1
            sentence.insert(pos_min[1], '[unused1]')
            # 分情况讨论，如果头实体的右侧是在尾实体右侧的左边的，那么两者相当于完全分开,done
            if pos_min[1] <= pos_max[0]:
                pos_max[0] = pos_max[0] + 1
                pos_max[1] = pos_max[1] + 1
                sentence.insert(pos_max[0], '[unused2]')
                pos_max[1] = pos_max[1] + 1
                sentence.insert(pos_max[1], '[unused3]')
            else:
                # 这是否两者没有完全分开，继续分情况讨论
                if pos_min[1] >= pos_max[1]:
                    # 如果头实体完全包裹尾实体则都不动,done
                    sentence.insert(pos_max[0], '[unused2]')
                    pos_max[1] = pos_max[1] + 1
                    sentence.insert(pos_max[1], '[unused3]')
                else:
                    # 说明头实体的一部份在尾实体的内部, done
                    pos_max[1] = pos_max[1] + 1
                    sentence.insert(pos_max[0], '[unused2]')
                    pos_max[1] = pos_max[1] + 1
                    sentence.insert(pos_max[1], '[unused3]')
                    
        # 整个text已经写完，然后开始进行编码，最大长度设置为clip中的77，超过这个长度直接截断
        eot_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        max_length, input_id_len, attention_mask_len = 77, 1, 1
        input_id, attention_mask = [49406,], [1,]
        tokenize_phrase_position = []
        find_phrase_flag = False
        for word in sentence:
            if any([phrase in word for phrase in item['grounding']]) or any([phrase.strip('.') in word.strip('.') for phrase in item['grounding']]):
                # 说明这是短语对应的位置
                find_phrase_flag = True
                phrase_position_start = input_id_len
            # 取出它的input_ids
            encoded_input_ids = self.processor.tokenizer(word)['input_ids'][1:-1]  
            input_id_len = input_id_len + len(encoded_input_ids)
            # 加入编码
            input_id.extend(encoded_input_ids)
            # 加入attention mask
            attention_mask.extend([1]*len(encoded_input_ids))
            attention_mask_len = attention_mask_len + len(encoded_input_ids)
            if find_phrase_flag:
                find_phrase_flag = False
                phrase_position_end = input_id_len
                # 加入检测到的短语的位置
                tokenize_phrase_position.append((phrase_position_start, phrase_position_end))
        # 插入结尾的标志符
        input_id.append(eot_token_id)
        input_id_len = input_id_len + 1
        attention_mask.append(1)
        attention_mask_len = attention_mask_len + 1
        assert input_id_len == attention_mask_len
        # 如果不足则padding至最大长度,从这里开始debug
        if len(input_id) < max_length:  
            input_id.extend([0] * (max_length - len(input_id)))
            attention_mask.extend([0] * (max_length - len(attention_mask)))
        # 长度过长则直接截断
        else:
            # 直接截断
            input_id = input_id[:max_length]
            attention_mask = attention_mask[:max_length]
            # 最后一位一定是结尾的位置
            input_id[-1] = eot_token_id
        # 整理短语的索引，避免超出
        for i in range(len(tokenize_phrase_position)-1, -1, -1):
            if tokenize_phrase_position[i][0] >= max_length:
                del tokenize_phrase_position[i]
            if tokenize_phrase_position[i][1] > max_length:
                tokenize_phrase_position[i][1] = max_length
        
        # 和论文一样，最长只采用6个
        tokenize_phrase_position = tokenize_phrase_position[:6]
    
        # 长度不足6，则扩充为6个
        if len(tokenize_phrase_position) < 6:
            tokenize_phrase_position.extend([(-1, -1)]*(6-len(tokenize_phrase_position)))
        
        input_id = torch.tensor(input_id).long().unsqueeze(0)   # (1, max_length)
        attention_mask = torch.tensor(attention_mask).long().unsqueeze(0)  # (1, max_length)
        phrase_position = torch.tensor(tokenize_phrase_position).unsqueeze(0)   #  (1, 6, 2)
        # 进行存储
        seq = [input_id, attention_mask, phrase_position]
        
        # load the original image，这里也是细粒度得到的图像，并且全部都resize成224大小的了
        image_ori = Image.open(os.path.join(self.pic_path_FineGrained_ori, self.data[index]['img_id'])).convert('RGB')
        img_ori = self.processor(images=image_ori, return_tensors='np')['pixel_values'].squeeze().tolist()  # 打成list

        # 得到的数据集中的原图,加上一层嵌套，具体为什么加，是和原来保持一样即可
        pic_ori = [img_ori]  
        # 原图中rcnn取出来的子图
        pic_ori_objects = aux_imgs_ori     # 已经用字典加了一层嵌套了
        
        np_pic2 = np.array(pic_ori).astype(np.float32)  # (1, 3, 224, 224)
        np_pic4 = np.array(pic_ori_objects).astype(np.float32)  # (3, 3, 128, 128)
        # 代表的是原图
        list_p2 = list(torch.tensor(np_pic2).unsqueeze(0)) #列表里面是[1,1,3,224,224]
        # 代表的是visual grounding的
        list_p4 = list(torch.tensor(np_pic4).unsqueeze(0)) #列表里面是[1,3,3,224,224]
        # 第一项肯定是他的标签label,然后是这个label对应的图像的真实索引，seq是tokenize之后处理好的每一项，原图，扩散图，原图vg,扩散图vg，四个分数，合并成一个列表传出去
        res = [img_id] + [data_id] + seq + list_p2  + list_p4

        return res  
    
    def collate_fn_sim(data):
        data = list(zip(*data))
        img_id = data[0]
        data_id = data[1]
        seqs = data[2:]
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))   # 所有的部分都打成batch
        return [img_id] + [data_id] + batch_seqs  #batch_seqs是一个列表长度为5，batch_seq[0]到batch_seq[4]分别为文本和图像的输入，input_id等等...
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        img_id = data[1]
        seqs = data[2:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            # print(seq)
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + [img_id] + batch_seqs

    def eval(self, pred_result, use_name=False):   # 这个模块其实就是画混淆矩阵的模块

        correct = 0
        total = len(self.data)
        correct_positive = 0  # 统计非None类别中positive的个数
        pred_positive = 0     # 统计预测结果为非negative的个数
        gold_positive = 0     # 统计groudtruth为positive的个数
        correct_category = np.zeros([31, 1])   # 各个类别的正确数量统计
        org_category = np.zeros([31, 1])   # 真实样本的无偏统计
        n_category = np.zeros([31, 1])     # 真实样本的negative统计
        data_with_pred_T = []   # 预测结果正确的样本记录
        data_with_pred_F = []   # 预测结果错误的样本记录
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]    # 默认采用这个，转化成了对应的id标号,也就是给'None'类别定为negative类别
                break
        y_pred = []   # 每一条样本的预测结果
        y_gt = []     # 每一条样本的groundtruth类别
        for i in range(total):
            y_pred.append(pred_result[i])   # 当前样本预测的结果
            y_gt.append(self.rel2id[self.data[i]['relation']])  # 当前样本的groundtruth的结果
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]  # 对应的id类别
                n_category[golden] += 1   # 统计样本的ground truth类别，在对应的id类别上+1
            data_with_pred = (str(self.data[i]) + str(pred_result[i]))  # 对应的样本和id标号上+1
            if golden == pred_result[i]:   # 判断是否一致
                correct += 1  # 一致则加1
                data_with_pred_T.append(data_with_pred)  # 记录这一组正确的类别
                if golden != neg:   # 如果不是negative类
                    correct_positive += 1   # 在positive的correct上+1
                    correct_category[golden] += 1
                else:
                    correct_category[0] += 1  # 否则加在第一个negative也就是None的类别上
            else:
                data_with_pred_F.append(data_with_pred)  # 不一致
            if golden != neg:
                gold_positive += 1
                org_category[golden] += 1
            else:
                org_category[0] += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)   # 预测为positive且正确的和预测为positive的(准确率)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)   # 预测为positive且正确的和groundtruth中为positive的(召回率)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)  # 记录就可以得到micro_f1
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))   # 返回的是最终的结果，预测正确中的各个类别，真实情况下的各个类别，真实情况下的None，以及预测正确和错误的样本标记
        return result, correct_category, org_category, n_category, data_with_pred_T, data_with_pred_F


class SentenceREDataset(data.Dataset):   # 自己定义的一套sentencedataset


    def __init__(self, text_path, pic_path, rel2id, tokenizer, sample_ratio, kwargs):


        super().__init__()
        self.text_path = text_path
        if 'train' in text_path:
            mode = 'train'
        elif 'val' in text_path:
            mode = 'val'
        else:
            mode = 'test'

        # get generated images path

        # self.pic_path_FineGrained_dif = '/home/thecharm/re/RE/diffusion_pic/{}/'.format(mode)
        self.pic_path_FineGrained_dif = pic_path.replace('img_org', 'diffusion_pic')   # 用diffsuion model生成的一系列图像
        # get original images path
        self.pic_path_FineGrained_ori = pic_path    # 它作为粗粒度的
        self.pic_path_CoarseGrained_ori = pic_path.replace('org', 'vg')    # 用visual grounding工具框出来的一系列图像，它作为细粒度

        # Load the text file
        self.data = []   # 用于存储整个数据文件的text文件
        f = open(text_path, encoding='UTF-8')  # 读取所有样本的text注释
        f_lines = f.readlines()
        for i1 in tqdm(range(len(f_lines))):
            line = f_lines[i1].rstrip()   # 把最右侧的换行符去掉
            if len(line) > 0:
                dic1 = eval(line)    # 把用字符串存储的样本，用字典给它切分进行存储，字典的键包括了'token, h, t, img_id, relation'，但是列表对应的数字下标索引是从0开始的
                self.data.append(dic1)
        f.close()
        self.rel2id = rel2id   # 分类类别转化成数字id
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(text_path, len(self.data),
                                                                                   len(self.rel2id)))

        # get the path of reflecting dict
        self.img_aux_path_dif = text_path.replace('ours_{}.txt'.format(mode), 'mre_dif_{}_dict.pth'.format(mode))
        self.img_aux_path_ori = text_path.replace('ours_{}.txt'.format(mode), 'mre_{}_dict.pth'.format(mode))
        # load the reflecting dict
        self.state_dict_dif = torch.load(self.img_aux_path_dif)   # 上面的这个是tensor，对应的是左上角和右下角的框在原图中的坐标索引，这个索引应该是提前和self.data中的索引对齐的，下面的也是
        self.state_dict_ori = torch.load(self.img_aux_path_ori)   # 下面的这个对应是原图中visual grounding得到的一系列路径，但是它的键其实是它的数字id标号，这一点是要值得注意的

        # get the path of correlation scores，只采用weak和strong的分数，分别是original和diffusion model生成的结果
        self.weak_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_weak.txt'.format(mode))
        self.strong_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_strong.txt'.format(mode))
        self.weak_dif = text_path.replace('ours_{}.txt'.format(mode), 'dif_{}_weight_weak.txt'.format(mode))
        self.strong_dif = text_path.replace('ours_{}.txt'.format(mode), 'dif_{}_weight_strong.txt'.format(mode))
        # load the correlation scores
        with open(self.weak_ori, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.weak_ori = {}   # 读取每一张图像中weak的分数，这里的所有分数中，它的索引应该是和self.data中的img_id是一一对应的
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.weak_ori[img_id_key] = score
        with open(self.strong_ori, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.strong_ori = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.strong_ori[img_id_key] = score
        with open(self.weak_dif, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.weak_dif = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.weak_dif[img_id_key] = score
        with open(self.strong_dif, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.strong_dif = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.strong_dif[img_id_key] = score

        self.tokenizer = tokenizer   # 这个是没有作用的，不用管，它在后面的过程中起到了分词的作用
        self.kwargs = kwargs
        self.transform = transforms.Compose([    # 这里设置了对图像进行一系列的变换，是用于diffsuion model中crop的图像
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.tokenizer_ = BertTokenizer.from_pretrained('/root/autodl-tmp/workspace/TMP_ACL2023/TMR/TMR-RE/bert-base-uncased')

        # get the path of phrases
        self.phrase_path = text_path.replace('ours_{}.txt'.format(mode), 'phrase_text_{}.json'.format(mode))
        f_grounding = open(self.phrase_path, 'r')
        self.phrase_data = json.load(f_grounding)   # 每一条样本检测出来的短语，但是它的键值是数字标号，这个标号也是提前和self.data是对齐的

        
    # 实际上dataset中一共得到了打乱后的self.data, self.state_dict_dif, self.state_dict_ori, self.phrase_data这几样，他们都是对齐的，而strong和weak的分数则是根据img_id进行读取的
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):    # 开始依次取出相应的样本，这里是按照shuffle进行读取的，并且读取的列表中提前用choice打乱过，所以这里的索引是对不上的，但是并不影响
        aux_imgs_dif = []     # 存储diffusion model生成的image
        aux_imgs_ori = []     # 存储original images，下面用于读取扩散模型生成的图像
        image_dif = Image.open(os.path.join(self.pic_path_FineGrained_dif, self.data[index]['img_id'])).convert('RGB')

        for i in range(3):   # 这里取3，表明visual grounding中取出的数据，只会用到三个，并且有时候还不足三个
            # get visual objects from generated images
            if len(self.state_dict_dif[index])>i:
                xy = self.state_dict_dif[index][i].tolist()
                aux_img_dif = image_dif.crop((xy[0], xy[1], xy[2], xy[3]))   # 裁剪的是左上角和右下角的坐标
                try:
                    img_features = self.transform(aux_img_dif).tolist()   # 对它进行resnet50的预处理
                    aux_img_dif = img_features
                    aux_imgs_dif.append(aux_img_dif)   # 处理完之后每一个图像得到的是一个列表，代表RGB三个通道
                except: 
                    a = 1 # do nothing and skip，操作不合理的话则直接跳过

            # get visual objects from original images
            if len(self.state_dict_ori[index])>i:    # 这里也是最多用3，也就是visual grounding的部分图像都是最多用3个
                image_ori_object = Image.open(os.path.join(self.pic_path_CoarseGrained_ori, 'crops/' + self.state_dict_ori[index][i])).convert('RGB')
                img_features_ori = self.transform(image_ori_object).tolist()
                aux_img_ori = img_features_ori
                aux_imgs_ori.append(aux_img_ori)   # 处理完之后每一个图像得到的是一个列表，代表RGB三个通道

        # padding zero tensor if less than 3
        for i in range(3 - len(aux_imgs_dif)):
            aux_imgs_dif.append(torch.zeros((3, 224, 224)).tolist())    # 不足的全部padding成0

        for i in range(3 - len(aux_imgs_ori)):
            aux_imgs_ori.append(torch.zeros((3, 224, 224)).tolist())    # 不足的全部padding成0

        assert len(aux_imgs_dif) == len(aux_imgs_ori) == 3

        item = self.data[index]    # 读取这条样本的文本数据
        item['grounding'] = self.phrase_data[str(index)] # phrase，读取这条样本的短语
        seq = list(self.tokenizer(item, **self.kwargs)) # 这里开始用分词函数来处理每一个text文本，转换成list来存储处理好的每一个文本
        img_id = item['img_id'] # img_id
        h = item['h'] # head entity
        t = item['t'] # tail entity
       

        # load the generated image，读取细粒度得到的图像
        image_dif = cv2.imread((os.path.join(self.pic_path_FineGrained_dif, self.data[index]['img_id'])))
        size = (224, 224)
        img_features_dif = cv2.resize(image_dif, size, interpolation=cv2.INTER_AREA)
        img_features_dif = torch.tensor(img_features_dif)
        img_features_dif = img_features_dif.transpose(1, 2).transpose(0, 1)
        img_dif = torch.reshape(img_features_dif, (3, 224, 224)).to(torch.float32).tolist()

        # load the original image，这里也是细粒度得到的图像，并且全部都resize成224大小的了
        image_ori = cv2.imread((os.path.join(self.pic_path_FineGrained_ori, self.data[index]['img_id'])))
        size = (224, 224)
        img_features_ori = cv2.resize(image_ori, size, interpolation=cv2.INTER_AREA)
        img_features_ori = torch.tensor(img_features_ori)
        img_features_ori = img_features_ori.transpose(1, 2).transpose(0, 1)
        img_ori = torch.reshape(img_features_ori, (3, 224, 224)).to(torch.float32).tolist()

        pic_dif = [img_dif]  # 得到的diffsuion model的原图，他们全部都是用list，也就是列表进行存储的
        pic_ori = [img_ori]  # 得到的数据集中的原图
        pic_dif_objects = aux_imgs_dif   # diffsuion中visual grounding取出来的子图，最多3个
        pic_ori_objects = aux_imgs_ori   # 原图中visual groudning取出来的子图，最多3个

        np_pic1 = np.array(pic_dif).astype(np.float32)  # (1, 3, 224, 224)
        np_pic2 = np.array(pic_ori).astype(np.float32)  # (1, 3, 224, 224)
        np_pic3 = np.array(pic_dif_objects).astype(np.float32)  # (3, 3, 224, 224)
        np_pic4 = np.array(pic_ori_objects).astype(np.float32)  # (3, 3, 224, 224)
        weight = [self.weak_ori[img_id], self.strong_ori[img_id], self.weak_dif[img_id], self.strong_dif[img_id]]  # 得到的原图和diffusion生成图的四个分数

        list_p1 = list(torch.tensor(np_pic1).unsqueeze(0))
        list_p2 = list(torch.tensor(np_pic2).unsqueeze(0))
        list_p3 = list(torch.tensor(np_pic3).unsqueeze(0))
        list_p4 = list(torch.tensor(np_pic4).unsqueeze(0))
       
        res = [self.rel2id[item['relation']]] + [img_id] + seq + list_p1 + list_p2 + list_p3 + list_p4 + [torch.tensor(weight).reshape(1,4)]

        return res  # label, seq1, seq2, ...,pic

    def collate_fn(data):
        data = list(zip(*data))
        
        labels = data[0]
        img_id = data[1]
        seqs = data[2:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            # print(seq)
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + [img_id] + batch_seqs

    def eval(self, pred_result, use_name=False):   # 这个模块其实就是画混淆矩阵的模块
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0  # 统计非None类别中positive的个数
        pred_positive = 0     # 统计预测结果为非negative的个数
        gold_positive = 0     # 统计groudtruth为positive的个数
        correct_category = np.zeros([31, 1])   # 各个类别的正确数量统计
        org_category = np.zeros([31, 1])   # 真实样本的无偏统计
        n_category = np.zeros([31, 1])     # 真实样本的negative统计
        data_with_pred_T = []   # 预测结果正确的样本记录
        data_with_pred_F = []   # 预测结果错误的样本记录
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]    # 默认采用这个，转化成了对应的id标号,也就是给'None'类别定为negative类别
                break
        y_pred = []   # 每一条样本的预测结果
        y_gt = []     # 每一条样本的groundtruth类别
        for i in range(total):
            y_pred.append(pred_result[i])   # 当前样本预测的结果
            y_gt.append(self.rel2id[self.data[i]['relation']])  # 当前样本的groundtruth的结果
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]  # 对应的id类别
                n_category[golden] += 1   # 统计样本的ground truth类别，在对应的id类别上+1
            data_with_pred = (str(self.data[i]) + str(pred_result[i]))  # 对应的样本和id标号上+1
            if golden == pred_result[i]:   # 判断是否一致
                correct += 1  # 一致则加1
                data_with_pred_T.append(data_with_pred)  # 记录这一组正确的类别
                if golden != neg:   # 如果不是negative类
                    correct_positive += 1   # 在positive的correct上+1
                    correct_category[golden] += 1
                else:
                    correct_category[0] += 1  # 否则加在第一个negative也就是None的类别上
            else:
                data_with_pred_F.append(data_with_pred)  # 不一致
            if golden != neg:
                gold_positive += 1
                org_category[golden] += 1
            else:
                org_category[0] += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)   # 预测为positive且正确的和预测为positive的(准确率)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)   # 预测为positive且正确的和groundtruth中为positive的(召回率)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)  # 记录就可以得到micro_f1
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))   # 返回的是最终的结果，预测正确中的各个类别，真实情况下的各个类别，真实情况下的None，以及预测正确和错误的样本标记
        return result, correct_category, org_category, n_category, data_with_pred_T, data_with_pred_F


def SentenceRELoader(text_path,  pic_path, rel2id, tokenizer,
                     batch_size, shuffle, subset_flag=False, subset_indices=None, num_workers=8, sample_ratio=1.0, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(text_path=text_path, pic_path=pic_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer,
                                sample_ratio=sample_ratio,
                                kwargs=kwargs)
    if subset_flag:
        dataset = data.Subset(dataset, subset_indices)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

def SimLoader(text_path, pic_path, pretrained_path, batch_size, subset_flag=False, subset_indices=None, num_workers=8,
              collate_fn=SimDataset.collate_fn_sim, **kwargs):
    
    dataset = SimDataset(text_path=text_path, pic_path=pic_path,
                        pretrained_path=pretrained_path, kwargs=kwargs)
    
    if subset_flag:
        dataset = data.Subset(dataset, subset_indices)
        shuffle=True
    else:
        shuffle=False
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,   # 这里的batch_size自己定义一下
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader
