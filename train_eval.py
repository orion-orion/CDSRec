# -*- coding: utf-8 -*-
import logging
import gc
import copy
from collections import defaultdict
import numpy as np
import tensorflow as tf
from data_utils import compute_rel_pos, split_A_and_B, get_ui_and_labels, align_A_and_B, get_eval_ui, get_predictions, \
    stack_index_and_pos


def train(model, train_dataloader, valid_dataloader, test_dataloader, args):   
    with tf.Session(graph=model.graph, config=model.config) as sess:
        sess.run(tf.global_variables_initializer())
        # Save all validation results in history (MRR10-A + MRR10-B)
        val_res_history = [0]
        for epoch in range(args.epochs):
            loss = 0
            step = 0
            for user_ids, sessions in train_dataloader:
                if args.method == "TiSASRec":
                    seqs, pos_samples, neg_samples = sessions 
                    _, l, = sess.run([model.train_op, model.loss], \
                        {model.input_seq: seqs, model.time_matrix: compute_rel_pos(seqs.shape[0], seqs.shape[1]), \
                            model.pos_samples: pos_samples, model.neg_samples: neg_samples, model.is_training: True})
                elif args.method == "CoNet":
                    items_A, items_B, neg_samples_A, neg_samples_B = sessions
                    ui_A, labels_A = get_ui_and_labels(user_ids, items_A, neg_samples_A, args.pad_int)
                    ui_B, labels_B = get_ui_and_labels(user_ids, items_B, neg_samples_B, args.pad_int)
                    ui_A, labels_A, ui_B, labels_B = \
                        align_A_and_B(ui_A, labels_A, ui_B, labels_B)    
                    _, l, = sess.run([model.train_op_joint, model.loss_joint], \
                    {model.input_A: ui_A, model.label_A: labels_A, \
                        model.input_B: ui_B, model.label_B: labels_B})
                elif args.method == "PINet":          
                    seqs_A, seqs_B, positions_A, positions_B, lens_A, lens_B, grounds_A, grounds_B = sessions
                    positions_A = stack_index_and_pos(positions_A)
                    positions_B = stack_index_and_pos(positions_B)                              
                    _, l = sess.run([model.train_op, model.loss], {model.seq_A: seqs_A, model.seq_B: seqs_B, \
                        model.pos_A: positions_A, model.pos_B: positions_B, model.len_A: lens_A, model.len_B: lens_B, \
                            model.ground_truth_A: grounds_A, model.ground_truth_B: grounds_B})
                loss += l
                step += 1
            gc.collect()
            logging.info('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch + 1, args.epochs, loss / step))
            if epoch % args.eval_interval == 0 and epoch > 10:
                val_metrics_A, val_metrics_B = evaluate(model, sess, valid_dataloader, args, mode="valid")
                if val_metrics_A["MRR10"] + val_metrics_B["MRR10"] > max(val_res_history):
                    evaluate(model, sess, test_dataloader, args, mode="test")   
                val_res_history.append(val_metrics_A["MRR10"] + val_metrics_B["MRR10"])


def evaluate(model, sess, dataloader, args, mode="valid"):
    eval_metrics_A = {
        "MRR1": 0.0, "MRR5": 0.0, "MRR10": 0.0, 
        "HR1": 0.0, "HR5": 0.0, "HR10": 0.0, 
        "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0,
    }
    eval_metrics_B = copy.copy(eval_metrics_A)
    
    num_samples_A, num_samples_B = 0, 0
    for user_ids, sessions in dataloader:
        if args.method == "TiSASRec":
            seqs, ground_truths, neg_samples = sessions
            predictions = sess.run(model.test_logits, \
            {model.input_seq: seqs, model.time_matrix: compute_rel_pos(seqs.shape[0], seqs.shape[1]), model.is_training: False})
            predictions_A, predictions_B, grounds_A, grounds_B, neg_samples_A, neg_samples_B = \
                split_A_and_B(predictions, ground_truths, neg_samples, method="TiSASRec", num_items_A=dataloader.num_items_A)
        elif args.method == "CoNet":
            grounds_A, grounds_B, neg_samples_A, neg_samples_B = sessions
            ui_A = get_eval_ui(user_ids, grounds_A, neg_samples_A)
            ui_B = get_eval_ui(user_ids, grounds_B, neg_samples_B)
            ui_preds_A = sess.run(model.logits_A_only, {model.input_A: ui_A})
            ui_preds_B = sess.run(model.logits_B_only, {model.input_B: ui_B})
            predictions_A = get_predictions(ui_A, ui_preds_A)
            predictions_B = get_predictions(ui_B, ui_preds_B)
        elif args.method == "PINet":   
            seqs_A, seqs_B, positions_A, positions_B, lens_A, lens_B, A_or_B, ground_truths, neg_samples = sessions            
            positions_A = stack_index_and_pos(positions_A)
            positions_B = stack_index_and_pos(positions_B)
            predictions_A, predictions_B = sess.run([model.logits_A, model.logits_B], {model.seq_A: seqs_A, model.seq_B: seqs_B, \
                model.pos_A: positions_A, model.pos_B: positions_B, model.len_A: lens_A, model.len_B: lens_B, \
                   model.ground_truth_A: ground_truths, model.ground_truth_B: ground_truths})
            predictions_A, predictions_B, grounds_A, grounds_B, neg_samples_A, neg_samples_B = \
                split_A_and_B((predictions_A, predictions_B), ground_truths, neg_samples, method="PINet", A_or_B=A_or_B)                 
            
        batch_eval_metrics_A = compute_eval_result(user_ids, predictions_A, grounds_A, neg_samples_A, k_list=[1, 5, 10], method=args.method)
        batch_eval_metrics_B = compute_eval_result(user_ids, predictions_B, grounds_B, neg_samples_B, k_list=[1, 5, 10], method=args.method)
        # The evaluation results are sumed up before averaging
        for key in eval_metrics_A.keys(): 
            eval_metrics_A[key] += batch_eval_metrics_A[key]
            eval_metrics_B[key] += batch_eval_metrics_B[key]            

        num_samples_A += grounds_A.shape[0]
        num_samples_B += grounds_B.shape[0]
        
    gc.collect()
    # Average the evaluation results
    for key in eval_metrics_A.keys():
        eval_metrics_A[key] /= num_samples_A
        eval_metrics_B[key] /= num_samples_B   

    if mode == "valid":
        logging.info('Valid:')
    else:
        logging.info('Test:')
    logging_evaluation_result(eval_metrics_A, domain="A")
    logging_evaluation_result(eval_metrics_B, domain="B")
    logging.info("")
    return eval_metrics_A, eval_metrics_B


def compute_eval_result(user_ids, predictions, ground_truths, neg_samples, k_list=[1, 5, 10], method="TiSASRec"):
    """ Returns MRR@k, HR@k and HDCG@k.
    """
    eval_metrics = {
        "MRR1": 0.0, "MRR5": 0.0, "MRR10": 0.0, 
        "HR1": 0.0, "HR5": 0.0, "HR10": 0.0, 
        "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0,
    }    
    def compute_eval_metrics(ground_item_rank):
        for k in k_list:
            if ground_item_rank <= k:
                eval_metrics["MRR%s" % k] += 1 / ground_item_rank
                eval_metrics["HR%s" % k] += 1
                eval_metrics["NDCG%s" % k] += 1 / np.log2(ground_item_rank + 1)
                
    if method == "CoNet":
        # For CoNet, each prediction corresponds to a unique user ID 
        for user_id, ground_truth, item_neg_samples in zip(user_ids, ground_truths, neg_samples):
            pred = predictions[user_id]
            pred_ground = pred[ground_truth] # predictions of ground truth item
            # How many negative samples have the predicted score larger than the ground truth item
            num_score_larger = 0
            for neg_sample in item_neg_samples:
                if pred[neg_sample] > pred_ground:
                    num_score_larger += 1
            # The rank of the ground truth item is num_score_larger + 1
            # If there is no negative sample whose predicted score is larger than the ground truth item,
            # then ground truth item is ranked 1                      
            compute_eval_metrics(ground_item_rank=num_score_larger + 1)
    else: 
        # For others method, each prediction corresponds to a evaluation sample
        # (different evaluation samples may correspond to the same user ID) 
        for pred, ground_truth, item_neg_samples in zip(predictions, ground_truths, neg_samples):
            pred_ground = pred[ground_truth]
            num_score_larger = np.sum(pred[item_neg_samples] > pred_ground)
            compute_eval_metrics(ground_item_rank=num_score_larger + 1)

    return eval_metrics


def logging_evaluation_result(eval_metrics, domain):
    logging.info('MRR-%s @1|5|10: %.4f\t%.4f\t%.4f' % (domain, eval_metrics["MRR1"], eval_metrics["MRR5"], eval_metrics["MRR10"]))
    logging.info('HR-%s @1|5|10: %.4f\t%.4f\t%.4f' % (domain, eval_metrics["HR1"], eval_metrics["HR5"], eval_metrics["HR10"]))
    logging.info('NDCG-%s @1|5|10: %.4f\t%.4f\t%.4f' % (domain, eval_metrics["NDCG1"], eval_metrics["NDCG5"], eval_metrics["NDCG10"]))