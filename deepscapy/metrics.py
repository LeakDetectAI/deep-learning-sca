import numpy as np
import pandas as pd


def topk_categorical_accuracy_np(k=5, normalize=False):
    def topk_acc(y_true, y_pred):
        n_objects = y_pred.shape[-1]
        topK = y_pred.argsort(axis=1)[:, -k:][:, ::-1]
        accuracies = np.zeros_like(y_true, dtype=bool)
        # y_true = np.argmax(y_true, axis=1)
        for i, top in enumerate(topK):
            accuracies[i] = y_true[i] in top
        accuracies = np.mean(accuracies)
        if normalize:
            minimum = k / n_objects
            accuracies = (accuracies - minimum) / (1.0 - minimum)
        return accuracies

    return topk_acc


def evaluate_model(scores, y_test, top1, top3, top5, top10):
    t1, t3, t5, t10 = top1(y_test, scores), top3(y_test, scores), top5(y_test, scores), top10(y_test, scores)
    return {"Top1": t1, "Top3": t3, "Top5": t5, "Top10": t10}


def calculate_mean_rank(y_true, final_scores):
    scores_df = pd.DataFrame(data=final_scores)
    final_ranks = scores_df.rank(ascending=False, axis=1)
    final_ranks = final_ranks.to_numpy(dtype='int32')
    predicted_ranks = np.zeros(shape=(y_true.shape[0]))
    for itr in range(y_true.shape[0]):
        true_label = y_true[itr]
        predicted_ranks[itr] = final_ranks[itr, true_label]
    return np.mean(predicted_ranks)
