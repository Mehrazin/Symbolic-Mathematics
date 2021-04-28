import os
import sys
import re
import argparse
import pickle
import torch
from logger import create_logger

class Config():
    def __init__(self):
        # Define dataset path
        self.exp_type = 'Prime_BWD'
        self.Dataset_dir = os.path.join(os.getcwd(), 'Dataset' , self.exp_type)
        self.train_dir = os.path.join(self.Dataset_dir, 'prim_bwd.train')
        self.test_dir = os.path.join(self.Dataset_dir, 'prim_bwd.test')
        self.valid_dir = os.path.join(self.Dataset_dir, 'prim_bwd.valid')

        # Experiment path
        self.dump_path = os.path.join(os.getcwd(), 'Dumped')
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        self.exp_id = self.get_exp_id()
        self.exp_dir = os.path.join(self.dump_path, str(self.exp_id))
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

        # Experimental setup
        self.n_words = 0
        self.task = 'prim_bwd'
        self.rewrite_functions = ''
        self.clean_prefix_expr = True
        self.env_seed = 0
        self.reload_size = 40000
        self.eos_index = 0
        self.pad_index = 1

        #Training configurations
        self.eval_only = False
        self.epoch_size = 300000
        self.batch_size = 10
        self.learning_rate = 0.0001
        self.clip_grad_norm = 5
        self.stopping_criterion = ''
        self.validation_metrics = 'valid_prim_bwd_acc'
        self.load_model = False
        self.model_path = os.path.join(os.getcwd(), 'trained', 'model.pth')
        self.reload_checkpoint = ''
        self.save_periodic = 100
        self.beam_eval = False
        self.eval_verbose = 0
        self.eval_verbose_print = False
        if self.load_model:
            assert os.path.exists(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configurations
        self.model_dim = 512
        self.num_head = 8
        self.forward_expansion = 4
        self.max_position = 4096
        self.num_enc_layer = 6
        self.num_dec_layer = 6
        self.share_inout_emb = True

    def get_exp_id(self):
        """
        Returns an integer as an experiment id.
        """
        dir_list = os.listdir(self.dump_path)
        if len(dir_list) == 0 :
            id = 1
        else :
            dir_list = [int(dir) for dir in dir_list]
            id = max(dir_list) + 1
        return id

    def get_logger(self):
        """
        creates a logger
        """
        command = ["python", sys.argv[0]]
        for x in sys.argv[1:]:
            if x.startswith('--'):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match('^[a-zA-Z0-9_]+$', x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = ' '.join(command)
        self.command = command + ' --exp_id "%s"' % self.exp_id

        # create a logger
        logger = create_logger(os.path.join(self.exp_dir, 'train.log'), rank=0)
        logger.info("============ Initialized logger ============")
        logger.info("\n".join("%s: %s" % (k, str(v))
                              for k, v in sorted(dict(vars(self)).items())))
        logger.info("The experiment will be stored in %s\n" % self.exp_dir)
        logger.info("Running command: %s" % command)
        logger.info("")
        return logger
