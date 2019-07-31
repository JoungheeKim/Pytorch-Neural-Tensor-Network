import pandas as pd
import numpy
from argparse import ArgumentParser
from data_loader import MyDataLoader
from torch import nn
import torch
import logging
import os
import tools
from pytorch_pretrained_bert import BertTokenizer, BertModel
from model import DocumentNTN
from trainer import Trainer


def build_parser():
    parser = ArgumentParser()

    ##Common option
    parser.add_argument("--device", dest="device", default="gpu")

    ##Loader option
    parser.add_argument("--train_path", dest="train_path", default="data/train.csv")
    parser.add_argument("--valid_path", dest="valid_path", default="data/test.csv")
    parser.add_argument("--dict_path", dest="dict_path", default="word2vec/1")
    parser.add_argument("--save_path", dest="save_path", default=None)
    parser.add_argument("--max_sent_len", dest="max_sent_len", default=10, type=int)
    parser.add_argument("--max_svo_len", dest="max_svo_len", default=100, type=int)

    ##Model option
    parser.add_argument("--tensor_dim", dest="tensor_dim", default=64, type=int)
    parser.add_argument("--hidden_size", dest="hidden_size", default=64, type=int)
    parser.add_argument("--atten_size", dest="atten_size", default=64, type=int)
    parser.add_argument("--n_layers", dest="n_layers", default=1, type=int)
    parser.add_argument("--dropout_p", dest="dropout_p", default=0.05, type=float)

    ##Train option
    parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int)
    parser.add_argument("--lr", dest="lr", default=0.0001, type=int)
    parser.add_argument("--early_stop", dest="early_stop", default=1, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=16, type=int)

    config = parser.parse_args()
    return config

def run(config):
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(config)

    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)

    if not config.save_path and config.dict_path:
        all_subdir = [int(s) for s in os.listdir(config.dict_path) if os.path.isdir(os.path.join(config.dict_path, str(s)))]
        max_dir_num = 0
        if all_subdir:
            max_dir_num = max(all_subdir)
        max_dir_num += 1
        config.save_path = os.path.join(config.dict_path, str(max_dir_num))
        os.mkdir(config.save_path)

    logging.basicConfig(filename=os.path.join(config.save_path, 'train_log'),
                        level=tools.LOGFILE_LEVEL,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(tools.CONSOLE_LEVEL)
    logging.getLogger().addHandler(console)

    logging.info("##################### Start Training")
    logging.debug(vars(config))

    ##load data loader
    logging.info("##################### Load DataLoader")
    loader = MyDataLoader(train_path=config.train_path,
                          valid_path=config.valid_path,
                          dict_path=config.dict_path,
                          batch_size=config.batch_size,
                          max_sent_len=config.max_sent_len,
                          max_svo_len=config.max_svo_len)

    train, valid, label_list = loader.get_train_valid()
    num_class = len(label_list)
    logging.info("##################### Train Dataset size : [" + str(len(train)) + "]")
    logging.info("##################### Valid Dataset size : [" + str(len(valid)) + "]")
    logging.info("##################### class size : [" + str(num_class) + "]")

    config.__setattr__("num_class", num_class)
    config.__setattr__("class_info", label_list)

    dict_size = loader.get_dict_size()
    word_vec_dim = loader.get_dict_vec_dim()
    embedding = loader.get_embedding()

    logging.info("##################### Load 'NTN attention' Model")
    model = DocumentNTN(dictionary_size=dict_size,
                        embedding_size=word_vec_dim,
                        tensor_dim=config.tensor_dim,
                        num_class=config.num_class,
                        hidden_size=config.hidden_size,
                        attention_size=config.atten_size,
                        n_layers=config.n_layers,
                        dropout_p=config.dropout_p,
                        device=config.device)

    model.set_embedding(embedding)
    model.to(config.device)

    crit = nn.NLLLoss()
    trainer = Trainer(model=model,
                      crit=crit,
                      config=config,
                      device=config.device)
    history = trainer.train(train, valid)
    return history

if __name__ == "__main__":
    ##load config files
    config = build_parser()
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")
    run(config)