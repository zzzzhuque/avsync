#coding=utf-8
import torch

class config(object):
    env = 'default' # visdom的环境
    augment = True # 是否做图像数据增强
    #model1 = 'audioNetwork'
    #model2 = 'videoNetwork'

    #train_data_root = './data/train' # 训练集存放路径
    #test_data_root = './data/test' # 测试集存放路径
    save_model_path = './checkpoints' # 保存训练过模型的路径

    batch_size = 128
    num_workers = 4 # 加载数据的线程数
    shuffle = True
    drop_last = True # 如果数据集大小不能被batch_size整除，设置为Ture则删除那个不完整的batch

    print_freq = 20 # 每N个batch打印一次信息

    #debug_file = '/tmp/debug' # 如果有这个文件，就debug

    max_epoch = 20 # 训练次数

    load_amodel_path = None # 如果要加载模型。要在这里改
    audiolr = 0.001 # 1e-2  1e-4
    audioMomentum = 0.9
    load_vmodel_path = None
    videolr = 0.001
    videoMomentum = 0.9

    use_gpu = True
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    #lr_decay = 0.95 # 当loss上升，调整lr=lr*lr_decay
    #weight_decay = 1e-4 # 损失函数


opt = config()
