import numpy as np
from sklearn import preprocessing
import h5py
# import numpy as np
import sortscore
import h5py
import matplotlib
import matplotlib.pyplot as plt
# import pandas as pd
# matplotlib.use('qt5agg')
import pandas as pd
# from data.database4.get_matrix import get_circ_dis_matrix
def get_scores_matrix(fd):

    min_max_scaler = preprocessing.MinMaxScaler()

# norms = np.linalg.norm(matrix, axis=1)
    scores_matrix = min_max_scaler.fit_transform(matrix)
    return scores_matrix
    # print(scores_matrix)


def main__():
    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    for fd in range(1, 6):
        # 准备预测矩阵
        # database1
        # # for key in f.keys():
        # # print(f[key].shape)
        # circrna_disease_matrix = f['/infor']
        # circrna_disease_matrix = np.array(circrna_disease_matrix)
        # database3
        circrna_disease_matrix = np.array(df)
        # database4
       

        prediction_matrix = np.zeros(np.shape(circrna_disease_matrix))
        prediction_matrix = get_scores_matrix(fd)
        roc_circrna_disease_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
        
            for line in f.readlines():
                line = line.strip('\n')  # 去除文本中的换行符
                line = line.strip().split('\t')
                for d_index in range(1, len(line)-1):
                    roc_circrna_disease_matrix[int(line[0])][int(line[d_index])] = 1
            #     fd.write(line[0] + "\t" + str(count) + "\n")
            # for j in range(len(new_circrna_disease_matrix)):
            #     f.write(str(j))
            #     for k in range(len(new_circrna_disease_matrix[0])):
            #         if new_circrna_disease_matrix[j][k] == 1:
            #             f.write('\t' + str(k))
            #     f.write('\n')
        # score_matrix = np.loadtxt("score_matrix_5fold" + str(fd) + ".txt",
        #                           delimiter=",")  # score_matrix = np.loadtxt("score_matrix_5fold_2.txt", delimiter = ",")# score_matrix = np.loadtxt("score_matrix_5fold_"+str(fd)+".txt", delimiter = ",")
        # for score_matrix_index in range(0, len(score_matrix)):
        #     computed_index_row = int(score_matrix_index / len(prediction_matrix[0]))
        #     computed_index_col = int(score_matrix_index - computed_index_row * len(prediction_matrix[0]))
        #     prediction_matrix[computed_index_row][computed_index_col] = score_matrix[score_matrix_index]
        #
        # test_matrix = np.loadtxt("./data_five_fold_adjust_parameters/test_index_order" + str(fd) + ".txt",
        #                          delimiter=",")  # test_matrix = np.loadtxt("./data_split_5fold/test_index_order2.txt", delimiter = ",")# test_matrix = np.loadtxt("./data_split_5fold/test_index_order"+str(fd)+".txt", delimiter = ",")
        # # new_circrna_disease_matrix = circrna_disease_matrix.copy()
        # # for index in range(0, len(test_matrix)):
        # #     new_circrna_disease_matrix[int(test_matrix[index][0]), int(test_matrix[index][1])] = 0
        # # roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        # valid_matrix = np.loadtxt("./data_five_fold_adjust_parameters/valid_index_order.txt", delimiter=",")
        # test_matrix = test_matrix + valid_matrix
        # roc_circrna_disease_matrix = test_matrix + circrna_disease_matrix
        # c = [0] * roc_circrna_disease_matrix.shape[0]
        # c = np.array(c).T
        # roc_circrna_disease_matrix[:, fd] = c
        # rel_matrix = new_circrna_disease_matrix
        # count = 0
        # for j in range(0, test_matrix.shape[0]):
        #     for k in range(0, test_matrix.shape[1]):
        #         if int(test_matrix[j][k]) == 1:
        #             test_matrix[j][k] == score_matrix[count]
        #             count += 1
        #
        # test_matrix[np.where(roc_matrix == 2)] = -20
        # sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(test_matrix, roc_matrix)
        roc_circrna_disease_matrix = roc_circrna_disease_matrix+circrna_disease_matrix
        zero_matrix = np.zeros((prediction_matrix.shape[0], prediction_matrix.shape[1]))
        score_matrix_temp = prediction_matrix.copy()
        score_matrix = score_matrix_temp + zero_matrix  # 标签矩阵等于预测矩阵加抹除关系的矩阵
        minvalue = np.min(score_matrix)  # 每列中的最小值
        score_matrix[np.where(roc_circrna_disease_matrix == 2)] = minvalue - 20  # ？
        sorted_circrna_disease_matrix, sorted_score_matrix = sortscore.sort_matrix(score_matrix,
                                                                                   roc_circrna_disease_matrix)

        tpr_list = []
        fpr_list = []
        recall_list = []
        precision_list = []
        accuracy_list = []
        F1_list = []
        for cutoff in range(sorted_circrna_disease_matrix.shape[0]):
            P_matrix = sorted_circrna_disease_matrix[0:cutoff + 1, :]
            N_matrix = sorted_circrna_disease_matrix[cutoff + 1:sorted_circrna_disease_matrix.shape[0] + 1, :]
            TP = np.sum(P_matrix == 1)
            FP = np.sum(P_matrix == 0)
            TN = np.sum(N_matrix == 0)
            FN = np.sum(N_matrix == 1)
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy = (TN + TP) / (TN + TP + FN + FP)
            # print(TP, FP, FN)
            F1 = (2 * TP) / (2 * TP + FP + FN)
            # print(F1)
            accuracy_list.append(accuracy)
            F1_list.append(F1)
        all_tpr.append(tpr_list)
        all_fpr.append(fpr_list)
        all_recall.append(recall_list)
        all_precision.append(precision_list)
        all_accuracy.append(accuracy_list)
        all_F1.append(F1_list)

    tpr_arr = np.array(all_tpr)
    fpr_arr = np.array(all_fpr)
    recall_arr = np.array(all_recall)
    precision_arr = np.array(all_precision)
    accuracy_arr = np.array(all_accuracy)
    F1_arr = np.array(all_F1)

    mean_cross_tpr = np.mean(tpr_arr, axis=0)  # axis=0
    mean_cross_fpr = np.mean(fpr_arr, axis=0)
    mean_cross_recall = np.mean(recall_arr, axis=0)
    mean_cross_precision = np.mean(precision_arr, axis=0)
    mean_cross_accuracy = np.mean(accuracy_arr, axis=0)
    # 计算此次五折的平均评价指标数值
    mean_accuracy = np.mean(np.mean(accuracy_arr, axis=1), axis=0)
    mean_recall = np.mean(np.mean(recall_arr, axis=1), axis=0)
    mean_precision = np.mean(np.mean(precision_arr, axis=1), axis=0)
    mean_F1 = np.mean(np.mean(F1_arr, axis=1), axis=0)
    print("accuracy:%.4f,recall:%.4f,precision:%.4f,F1:%.4f" % (mean_accuracy, mean_recall, mean_precision, mean_F1))

    # print("%.4f"%2*mean_recall*mean_precision/(mean_recall+mean_precision))
    roc_auc = np.trapz(mean_cross_tpr, mean_cross_fpr)
    AUPR = np.trapz(mean_cross_precision, mean_cross_recall)

    

    print(roc_auc, AUPR)
    plt.plot(mean_cross_fpr, mean_cross_tpr, label='mean ROC=%0.4f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    # plt.savefig("DRGCNCDA.png")
    print("runtime over, now is :")
    # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    plt.show()


if __name__ == '__main__':
    main__()