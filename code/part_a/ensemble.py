import numpy as np
from utils import *
from part_a.matrix_factorization import update_u_z


def random_with_replacement(train_data):
    """
    :param train_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :return:
    """
    sample = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    for t in range(len(train_data["question_id"])):
        i = np.random.choice(len(train_data["question_id"]), 1)[0]
        sample["user_id"].append(train_data["user_id"][i])
        sample["question_id"].append(train_data["question_id"][i])
        sample["is_correct"].append(train_data["is_correct"][i])
    return sample


def simple_als(train_data, k, lr, num_iteration):

    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    for t in range(num_iteration):
        update_u_z(train_data, lr, u, z)

    mat = np.dot(u, z.T)
    return mat


def sparse_matrix_average_predictions(data, matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8, matrix9, matrix10, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix1: 2D matrix
    :param matrix2: 2D matrix
    :param matrix3: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        avg_pred = (matrix1[u,q]+matrix2[u,q]+matrix3[u,q]+matrix4[u,q]+matrix5[u,q]+matrix6[u,q]+matrix7[u,q]+matrix8[u,q]+matrix9[u,q]+matrix10[u,q]) / 10

        if avg_pred >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)

    return predictions


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    iteration = 500000
    k = 50

    data_1 = random_with_replacement(train_data)
    data_2 = random_with_replacement(train_data)
    data_3 = random_with_replacement(train_data)
    data_4 = random_with_replacement(train_data)
    data_5 = random_with_replacement(train_data)
    data_6 = random_with_replacement(train_data)
    data_7 = random_with_replacement(train_data)
    data_8 = random_with_replacement(train_data)
    data_9 = random_with_replacement(train_data)
    data_10 = random_with_replacement(train_data)

    matrix_1 = simple_als(data_1, k, lr, iteration)
    matrix_2 = simple_als(data_2, k, lr, iteration)
    matrix_3 = simple_als(data_3, k, lr, iteration)
    matrix_4 = simple_als(data_4, k, lr, iteration)
    matrix_5 = simple_als(data_5, k, lr, iteration)
    matrix_6 = simple_als(data_6, k, lr, iteration)
    matrix_7 = simple_als(data_7, k, lr, iteration)
    matrix_8 = simple_als(data_8, k, lr, iteration)
    matrix_9 = simple_als(data_9, k, lr, iteration)
    matrix_10 = simple_als(data_10, k, lr, iteration)

    val_pred = sparse_matrix_average_predictions(
        val_data, matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9, matrix_10)
    test_pred = sparse_matrix_average_predictions(
        test_data, matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7, matrix_8, matrix_9, matrix_10)

    val_acc = evaluate(val_data, val_pred)
    test_acc = evaluate(test_data, test_pred)
    print("Validation Accuracy: {}".format(val_acc))
    print("Test Accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main()
