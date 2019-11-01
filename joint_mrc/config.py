SEQ_MAXLEN = 512
ANS_MAXLEN = 80
DROPOUT = 0.1
N_LAYERS = 12
LR = 2e-5
BATCH_SIZE = 6
EPOCHS = 6
LOSS_WEIGHTS = [0.04, 0.48, 0.48]
FUSE_WEIGHTS = [0.98, 0.02]
MONITOR = 'val_loss'
PRE_MODEL = 'ROBERTA-WWM'
DATA_DIR = './datasets'
MODEL_DIR = './models'
BERT_DIR = './models/chinese_roberta_wwm_ext_L-12_H-768_A-12'
VOCAB_PATH = './models/vocab.txt'
PROCESSED_PATH = './datasets/processed.pkl'
RETRIEVE_DIR = './models/retrieve'
FIRST_GPU = False
POSITIVE_FIRST = True

class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            
    def add_argument(self, key, value):
        self.__setattr__(key, value)
        
"""参数配置
seq_maxlen：输入序列的最大长度，包括问题
ans_maxlen：答案的最大长度
dropout：每一层的失活率
n_layers：transformer的层数
lr：初始学习率
batch_size：训练、评估、推断的批大小
epochs：训练轮数
loss_weights：检索任务、答案头、答案尾的损失权重
fuse_weights：检索分数与答案分数的融合权重
monitor：模型选择指标
pre_model：预训练模型名称
data_dir：数据集目录
model_dir：模型存放目录
bert_dir：预训练模型参数文件目录
vocab_path：词典文件路径
processed_path：预处理文件路径
retrieve_dir: 第一阶段检索模型目录
first_gpu: 使用多卡并行训练时，是否使用第一块GPU，
           如果不使用则第一块GPU只做梯度汇总用途
positive_first: 是否参照正样本采样，如果是则每一轮
                包含所有正样本，负样本随机采样，如果
                否则参照数量较少的那一类做均衡采样
"""
config = Config(
    seq_maxlen=SEQ_MAXLEN,
    ans_maxlen=ANS_MAXLEN,
    dropout=DROPOUT,
    n_layers=N_LAYERS,
    lr=LR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    loss_weights=LOSS_WEIGHTS,
    fuse_weights=FUSE_WEIGHTS,
    monitor=MONITOR,
    pre_model=PRE_MODEL,
    data_dir=DATA_DIR,
    model_dir=MODEL_DIR,
    bert_dir=BERT_DIR,
    vocab_path=VOCAB_PATH,
    processed_path=PROCESSED_PATH,
    retrieve_dir=RETRIEVE_DIR,
    first_gpu=FIRST_GPU,
    positive_first=POSITIVE_FIRST,
    )