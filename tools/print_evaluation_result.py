from sklearn import metrics


def print_evaluation_result(target_Y, predict_Y):
    print('Accuracy: ', metrics.accuracy_score(target_Y, predict_Y))
    print('Confusion Metric: \n', metrics.confusion_matrix(target_Y,predict_Y))
    print('Micro Precision/Recall/F-score: ', metrics.precision_recall_fscore_support(target_Y, predict_Y, average='micro'))
    print('Macro Precision/Recall/F-score: ', metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro'))
    # print('Each-class Precision/Recall/F-score: \n', np.matrix(metrics.precision_recall_fscore_support(target_Y, predict_Y, average=None)))
    return metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')[2], \
           metrics.accuracy_score(target_Y, predict_Y)
    # return f1_score and acc
