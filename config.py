import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 72, 128  # input image size
        self.batch_size = 32  # train batch size
        self.test_batch_size = 8  # test batch size
        self.num_boxes = 12  # max number of bounding boxes in each frame

        self.frame_num = 32

        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = False
        self.device_list = "0"  # id list of gpus used for training

        # Dataset
        assert (dataset_name == 'volleyball')
        self.dataset_name = dataset_name

        if dataset_name == 'volleyball':
            self.data_path = 'data'  # data path for the volleyball dataset
            self.train_seqs = [45, 46, 49, 41, 52, 44]  # video id list of train set
            self.test_seqs = [51, 50]  # video id list of test set

        # Backbone 
        self.backbone = 'inv3'
        self.crop_size = 5, 5  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056  # output feature map channel of backbone

        # Activity Action
        self.num_activities = 8  # number of activity categories

        # Sample
        self.num_frames = 6
        self.num_before = 4
        self.num_after = 2

        # GCN
        self.num_features_boxes = 512
        self.num_features_relation = 256
        self.num_graph = 16  # number of graphs
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 5  # number of GCN layers
        self.tau_sqrt = False
        self.pos_threshold = 0.2  # distance mask threshold in position relation

        # GAT
        self.n_hid = 16
        self.n_heads = 8
        self.dropout = 0.5
        self.alpha = 0.01

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  # initial learning rate
        self.lr_plan = {41: 1e-4, 81: 5e-5, 121: 1e-5}  # change learning rate in these epochs
        self.train_dropout_prob = 0.3  # dropout probability
        self.weight_decay = 0  # l2 weight decay

        self.max_epoch = 150  # max training epoch
        self.test_interval_epoch = 2

        # Exp
        self.training_stage = 1  # specify stage1 or stage2
        self.stage1_model_path = ''  # path of the base model, need to be set in stage2
        self.test_before_train = False
        self.exp_note = 'GroupActivity'
        self.exp_name = None

    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s_stage%d]<%s>' % (self.exp_note, self.training_stage, time_str)
        print(self.exp_name)
        self.result_path = './result/%s' % self.exp_name
        print(self.result_path)
        self.log_path = './result/%s/log.txt' % self.exp_name
        print(self.result_path)
        if need_new_folder:
            os.mkdir(self.result_path)
