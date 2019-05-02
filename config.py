#coding=utf-8
import torch

class config(object):
    env = 'default' # visdom的环境
    #model1 = 'audioNetwork'
    #model2 = 'videoNetwork'

    #train_data_root = './data/train' # 训练集存放路径
    #test_data_root = './data/test' # 测试集存放路径
    load_model_path = './checkpoints' # 加载预训练模型，为none则不加载

    batch_size = 4
    num_workers = 4 # 加载数据的线程数
    shuffle = True
    #print_freq = 20 # 每N个batch打印一次信息

    #debug_file = '/tmp/debug' # 如果有这个文件，就debug

    max_epoch = 1 # 训练次数

    audiolr = 0.001
    audioMomentum = 0.9
    videolr = 0.001
    videoMomentum = 0.9

    use_gpu = True
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    #lr_decay = 0.95 # 当loss上升，调整lr=lr*lr_decay
    #weight_decay = 1e-4 # 损失函数


opt = config()
