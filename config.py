import os

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
        self.task = 'prim_bwd'
        self.rewrite_functions = ''
        self.clean_prefix_expr = True
        self.env_seed = 0
        self.reload_size = 40000
        self.eos_index = 0
        self.pad_index = 1

        #Training configurations
        self.epoch_size = 300000
        self.batch_size = 10
        self.learning_rate = 0.0001


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
