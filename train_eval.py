import logging
import gc
import copy
import numpy as np
import tensorflow as tf
from data_utils import compute_rel_pos, split_A_and_B


def train(model, train_dataloader, valid_dataloader, test_dataloader, args):   
    with tf.Session(graph=model.graph, config=model.config) as sess:
        sess.run(tf.global_variables_initializer())
        # Save all validation results in history (MRR10-A + MRR10-B)
        val_res_history = [0]
        for epoch in range(args.epochs):
            loss = 0
            step = 0
            for _, sessions in train_dataloader:
                if args.method == "TiSASRec":
                    seqs, pos_samples, neg_samples = sessions 
                    _, l, = sess.run([model.train_op, model.loss], \
                        {model.input_seq: seqs, model.time_matrix: compute_rel_pos(seqs.shape[0], seqs.shape[1]), \
                            model.pos_samples: pos_samples, model.neg_samples: neg_samples, model.is_training: True})
                loss += l
                step += 1
            gc.collect()
            logging.info('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch + 1, args.epochs, loss / step))
            if epoch % args.eval_interval == 0: 
                val_metrics_A, val_metrics_B = evaluate(model, sess, valid_dataloader, args, mod="valid")
                if val_metrics_A["MRR10"]  + val_metrics_B["MRR10"] > max(val_res_history):
                    evaluate(model, sess, test_dataloader, args, mod="test")   
                val_res_history.append(val_metrics_A["MRR10"]  + val_metrics_B["MRR10"])  


def evaluate(model, sess, dataloader, args, mod="valid"):
    eval_metrics_A = {
        "MRR1": 0.0, "MRR5": 0.0, "MRR10": 0.0, 
        "HR1": 0.0, "HR5": 0.0, "HR10": 0.0, 
        "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0,
    }
    eval_metrics_B = copy.copy(eval_metrics_A)
    
    num_seqs_A, num_seqs_B = 0, 0
    for _, sessions in dataloader:
        if args.method == "TiSASRec":
            seqs, ground_truths, neg_samples = sessions
            predictions = sess.run(model.test_logits, \
            {model.input_seq: seqs, model.time_matrix: compute_rel_pos(seqs.shape[0], seqs.shape[1]), model.is_training: False})
            predictions_A, predictions_B, grounds_A, grounds_B = split_A_and_B(predictions, ground_truths, dataloader.num_items_A)

        batch_eval_metrics_A = compute_eval_result(predictions_A, grounds_A, neg_samples, k_list=[1, 5, 10])
        batch_eval_metrics_B = compute_eval_result(predictions_B, grounds_B, neg_samples, k_list=[1, 5, 10])
        # The evaluation results are sumed up before averaging
        for key in eval_metrics_A.keys(): 
            eval_metrics_A[key] += batch_eval_metrics_A[key]
            eval_metrics_B[key] += batch_eval_metrics_B[key]            

        num_seqs_A += grounds_A.shape[0]
        num_seqs_B += grounds_B.shape[0]

        
    gc.collect()
    # Average the evaluation results
    for key in eval_metrics_A.keys():
        eval_metrics_A[key] /= num_seqs_A
        eval_metrics_B[key] /= num_seqs_B   

    if mod == "valid":
        logging.info('Valid:')
    else:
        logging.info('Test:')
    logging_evaluation_result(eval_metrics_A, domain="A")
    logging_evaluation_result(eval_metrics_B, domain="B")
    logging.info("")
    return eval_metrics_A, eval_metrics_B


def compute_eval_result(predictions, ground_truths, neg_samples, k_list):
    """ Returns MRR@k, HR@k and HDCG@k.
    """
    eval_metrics = {
        "MRR1": 0.0, "MRR5": 0.0, "MRR10": 0.0, 
        "HR1": 0.0, "HR5": 0.0, "HR10": 0.0, 
        "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0,
    }
    for pred, ground_truth, item_neg_samples in zip(predictions, ground_truths, neg_samples):
        pred_ground = pred[ground_truth] # predictions of ground truth item
        # How many negative samples have the predicted score larger than the ground truth item
        score_larger = (pred[item_neg_samples] > (pred_ground))
        # The rank of the ground truth item is its predictied score + 1
        # If there is no negative sample whose predicted score is larger than the ground truth item,
        # then ground truth item is ranked 1  
        ground_item_rank = np.sum(score_larger) + 1   
        for k in k_list:
            if ground_item_rank <= k:
                eval_metrics["MRR%s" % k] += 1 / ground_item_rank
                eval_metrics["HR%s" % k]  += 1
                eval_metrics["NDCG%s" % k] += 1 / np.log2(ground_item_rank + 1)
    return eval_metrics


def logging_evaluation_result(eval_metrics, domain):
    logging.info('MRR-%s @1|5|10: %.4f\t%.4f\t%.4f' % (domain, eval_metrics["MRR1"], eval_metrics["MRR5"], eval_metrics["MRR10"]))
    logging.info('HR-%s @1|5|10: %.4f\t%.4f\t%.4f' % (domain, eval_metrics["HR1"], eval_metrics["HR5"], eval_metrics["HR10"]))
    logging.info('NDCG-%s @1|5|10: %.4f\t%.4f\t%.4f' % (domain, eval_metrics["NDCG1"], eval_metrics["NDCG5"], eval_metrics["NDCG10"]))