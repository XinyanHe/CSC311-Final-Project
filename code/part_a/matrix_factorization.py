from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u[n] = u[n] + lr * (c - np.dot(u[n].T, z[q])) * z[q]
    z[q] = z[q] + lr * (c - np.dot(u[n].T, z[q])) * u[n]

    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    train_loss_lst = []
    val_loss_lst = []
    for t in range(num_iteration):
        update_u_z(train_data, lr, u, z)
        # if t % 2000 == 0:
        #     train_loss = squared_error_loss(train_data, u, z)
        #     val_loss = squared_error_loss(val_data, u, z)
        #     train_loss_lst.append(train_loss)
        #     val_loss_lst.append(val_loss)

    mat = np.dot(u, z.T)

    return mat, train_loss_lst, val_loss_lst


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print(" ==================== SVD ==================== ")

    # Try out different k and select the best k using validation set
    k_list = [5, 6, 7, 8, 9, 10, 11, 12]
    acc_list = []
    max_acc = 0
    best_k = 0
    for k in k_list:
        matrix = svd_reconstruct(train_matrix, k)
        acc = sparse_matrix_evaluate(val_data, matrix)
        acc_list.append(acc)
        if acc > max_acc:
            max_acc = acc
            best_k = k

    # Report final validation and test accuracy
    matrix = svd_reconstruct(train_matrix, best_k)
    val_acc = sparse_matrix_evaluate(val_data, matrix)
    test_acc = sparse_matrix_evaluate(test_data, matrix)
    print("The best k is {}, with validation accuracy {} and test accuracy {}"
          .format(best_k, val_acc, test_acc))

    print("\n ==================== ALS ==================== ")

    # Train with ALS
    lr = 0.01
    num_iteration = 500000
    k = 50
    matrix, train_loss_lst, val_loss_lst = als(
        train_data, val_data, k, lr, num_iteration)

    # Plot training and validation squared-error losses
    # iter_lst = np.arange(0, 500000, 2000)
    # plt.figure(figsize=(10, 7))
    # plt.rcParams.update({'font.size': 12})
    # plt.plot(iter_lst, train_loss_lst, label="Train")
    # plt.plot(iter_lst, val_loss_lst, label="Validation")
    # plt.xlabel("Iteration")
    # plt.ylabel("Squared-Error Loss")
    # plt.legend()
    # plt.show()

    # Report final validation and test accuracy
    val_acc = sparse_matrix_evaluate(val_data, matrix)
    test_acc = sparse_matrix_evaluate(test_data, matrix)
    print("Validation Accuracy: ", val_acc)
    print("Test Accuracy: ", test_acc)


if __name__ == "__main__":
    main()
