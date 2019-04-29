#coding=utf-8
class config(object):
    env = 'default' # visdom的环境
    model1 = 'audionetwork'
    model2 = 'visualnetwork'

    train_data_root = './data/train' # 训练集存放路径
    test_data_root = './data/test' # 测试集存放路径
    load_model_path = './checkpoints/model.pth' # 加载预训练模型，为none则不加载

    batch_size = 256
    use_gpu = True
    num_workers = 4 # 加载数据的线程数
    print_freq = 20 # 每N个batch打印一次信息

    debug_file = '/tmp/debug' # 如果有这个文件，就debug

    max_epoch = 10 # 训练次数

    audiolr = 0.1
    audioMomentum = 0.9
    videolr = 0.1
    videoMomentum = 0.9

    lr_decay = 0.95 # 当loss上升，调整lr=lr*lr_decay
    weight_decay = 1e-4 # 损失函数

    frames = 5
