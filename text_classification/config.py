# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 14:15
# @Author  : chile
# @Email   : realchilewang@foxmail.com
# @File    : config.py
# @Software: PyCharm


class Config(object):
    def __init__(self):
        self.base_dir = '../data/text_classification/'  # 数据路径
        self.save_model = self.base_dir + 'Savemodel/'  # 模型路径
        self.result_file = 'result/'
        self.label_list = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']

        self.warmup_proportion = 0.05
        self.use_bert = True
        self.pretrainning_model = 'nezha'
        self.embed_dense = 512

        self.decay_rate = 0.5  # 学习率衰减参数

        self.train_epoch = 20  # 训练迭代次数

        self.learning_rate = 1e-4  # 下接结构学习率
        self.embed_learning_rate = 5e-5  # 预训练模型学习率

        if self.pretrainning_model == 'roberta':
            # model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_roberta_base/'  # 中文roberta-base
            model = '../../autodl-tmp/pretrain_models/chinese-roberta-wwm-ext/'  # 中文roberta-base
        elif self.pretrainning_model == 'nezha':
            # model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_nezha_base/'  # 中文nezha-base
            model = '../../autodl-tmp/pretrain_models/nezha-cn-base/'  # 中文nezha-base
        else:
            raise KeyError('albert nezha roberta bert bert_wwm is need')
        self.cls_num = 10 #dev和train中文本共包含10个类别 '教育', '时政', '体育', '家居', '游戏', '科技', '娱乐', '房产', '财经', '时尚'
        self.sequence_length = 512
        self.batch_size = 16

        self.model_path = model

        self.bert_file = model + 'pytorch_model.bin'
        # self.bert_config_file = model + 'bert_config.json'
        self.bert_config_file = model + 'config.json'
        self.vocab_file = model + 'vocab.txt'

        self.use_origin_bert = 'ori'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'weight':初始化12*1向量
        self.is_avg_pool = 'mean'  # dym, max, mean, cls
        self.model_type = 'bilstm'  # bilstm; bigru

        self.rnn_num = 2
        self.flooding = 0
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        # 模型预测路径
        self.checkpoint_path = "/home/wangzhili/chile/tsing_hua_code/data/Savemodel/runs_1/1624196649/model_0.9082_0.9082_0.9082_167.bin"
        self.checkpoint_path = "/home/wangzhili/chile/tsing_hua_code/data/Savemodel/runs_0/1624199990/model_dist.bin"

        """
        实验记录
        """
