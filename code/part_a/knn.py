from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    k_list = [1, 6, 11, 16, 21, 26]
    acc_user_list = []
    acc_item_list = []
    max_acc_user = 0
    max_acc_item = 0
    k_best_user = 0
    k_best_item = 0

    for k in k_list:
        acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_user_list.append(acc_user)
        acc_item_list.append(acc_item)
        if acc_user > max_acc_user:
            max_acc_user = acc_user
            k_best_user = k
        if acc_item > max_acc_item:
            max_acc_item = acc_item
            k_best_item = k

    # Plot the validation accuracy as a function of k
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    plt.plot(k_list, acc_user_list, marker='o', label="By User")
    plt.plot(k_list, acc_item_list, marker='o', label="By Item")
    plt.xlabel("K (Number of Nearest Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

    # Report k* and the test accuracy
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, k_best_user)
    print("\nKNN impute by user:")
    print("k* is {} with test accuracy {}".format(k_best_user, test_acc_user))

    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_best_item)
    print("\nKNN impute by item:")
    print("k* is {} with test accuracy {}".format(k_best_item, test_acc_item))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
