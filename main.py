#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import argparse
import os
import logging
import numpy as np
import tensorflow as tf
from dataset import CDSDataSet, Dataloader
from tisasrec.tisasrec_model import TiSASRec
from train_eval import train


def arg_parse():
    parser = argparse.ArgumentParser()
    
    # Dataset part
    parser.add_argument("--dataset", type=str, default="Food-Kitchen",
                        help="dataset")
    parser.add_argument("--seed", type=int, default=42, help="random seed")                        
    parser.add_argument("--load_prep", dest="load_prep", action="store_true", default=False, help=\
        "Whether need to load preprocessed the data. If you want to load preprocessed data, add it") 
    parser.add_argument("--pad_int", type=int, default=0, help="padding on the session")

    # Training part
    parser.add_argument("--method", type=str, default="PINet", help="dataset, possible are \
        `TiSASRec`,`CoNet`, `PINet`, `MIFN`")
    parser.add_argument("--log_dir", type=str, default="log", help="directory of logs")
    parser.add_argument("--cuda", type=bool, default=tf.test.is_gpu_available())
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--epochs", type=int, default=60, help="number of training epochs")
    parser.add_argument("--optimizer", type=str, default="Adam", help="type of the optimizer. possible are \
        `Adam`,`RMSProp`, `AdaGrad`, `SGD`")    
    parser.add_argument("--lr", type=float, default=0.001, help="applies to Adam.")
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")     
    parser.add_argument("--eval_interval", type=int, default=1, help="interval of evalution")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="dropout rate")        
    
    args = parser.parse_args()
    return args


def seed_everything(args):
    os.environ["PYTHONHASHSEED"] = str(args.seed)    
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)    
    

def init_logger(args):
    log_file = os.path.join(args.log_dir, args.dataset + ".log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    
def main():
    args = arg_parse()

    seed_everything(args)

    init_logger(args)

    train_dataset = CDSDataSet(args.dataset, method=args.method, mode="train", pad_int=args.pad_int, load_prep=args.load_prep)
    valid_dataset = CDSDataSet(args.dataset, method=args.method, mode="valid", pad_int=args.pad_int, load_prep=args.load_prep)
    test_dataset = CDSDataSet(args.dataset, method=args.method, mode="test", pad_int=args.pad_int, load_prep=args.load_prep)

    train_dataloader = Dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = Dataloader(valid_dataset , batch_size=args.batch_size, shuffle=True)
    test_dataloader = Dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_items = train_dataset.num_items_A + train_dataset.num_items_B
    if args.method == "TiSASRec":       
        # max_seq_len is for positional encoding in TiSASRec
        max_seq_len = max(train_dataset.max_seq_len, valid_dataset.max_seq_len, test_dataset.max_seq_len)
        model = TiSASRec(num_items, max_seq_len, args)
    
    train(model, train_dataloader, valid_dataloader, test_dataloader, args)


if __name__ == "__main__":
    main()