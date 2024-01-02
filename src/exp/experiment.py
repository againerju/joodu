
import sys
import os
import datetime
from pytorch_lightning.loggers import TensorBoardLogger


def get_date_and_time_string():

    return datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    

def create_experiment_dir_name(debug=False):

    out = ""
    out += get_date_and_time_string()

    if debug:
        out += "_debug"
    
    return out


class Logger(object):
    def __init__(self, log_dir, log_name="log.txt"):
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, log_name)
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    # write a function which stops the logger from logging
    def close(self):
        self.log.close()
        sys.stdout = sys.stdout.terminal


def setup_logger(project_root, task="traj_pred", model_name="001_enc_dec"):
    exp_root = os.path.join(project_root, "experiments", task)
    exp_name = create_experiment_dir_name()
    tb_logger = TensorBoardLogger(
        save_dir=exp_root, name=model_name, version=exp_name)
    cmd_logger = Logger(tb_logger.log_dir)
    sys.stdout = cmd_logger
    print("\nCreating experiment directory for {}: {}".format(task, tb_logger.log_dir))
    return tb_logger, cmd_logger
