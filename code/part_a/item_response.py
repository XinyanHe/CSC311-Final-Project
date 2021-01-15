from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    num_observed = len(data["is_correct"])
    log_lklihood = 0.
    for k in range(num_observed):
        c = data["is_correct"][k]
        i = data["user_id"][k]
        j = data["question_id"][k]
        sig = sigmoid(theta[i] - beta[j])
        log_lklihood += c * np.log(sig) + (1 - c) * np.log(1 - sig)
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    num_observed = len(data["is_correct"])
    for k in range(num_observed):
        c = data["is_correct"][k]
        i = data["user_id"][k]
        j = data["question_id"][k]
        theta[i] -= lr * (- c + sigmoid(theta[i] - beta[j]))
        beta[j] -= lr * (c - sigmoid(theta[i] - beta[j]))
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst, train_acc_lst)
    """
    theta = np.zeros(542)
    beta = np.zeros(1774)

    train_neg_lld_lst = []
    train_lld_lst = []
    val_neg_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data=data, theta=theta, beta=beta)
        train_lld = (-1)* train_neg_lld
        val_neg_lld = neg_log_likelihood(data=val_data, theta=theta, beta=beta)
        val_lld = (-1)* val_neg_lld
        train_neg_lld_lst.append(train_neg_lld)
        train_lld_lst.append(train_lld)
        val_neg_lld_lst.append(val_neg_lld)
        val_lld_lst.append(val_lld)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        print("NLLK: {} \t Score: {}".format(train_neg_lld, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, train_neg_lld_lst, val_neg_lld_lst,train_lld_lst,val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Train IRT
    lr = 0.02
    num_iteration = 30
    iter_lst = np.arange(num_iteration)
    theta, beta, train_neg_lld_lst, val_neg_lld_lst, train_lld_lst, val_lld_lst\
        = irt(train_data, val_data, lr, num_iteration)

    # Plot training and validation neg_log_likelihood as a function of iterations
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    plt.plot(iter_lst, train_neg_lld_lst, label="Train")
    plt.plot(iter_lst, val_neg_lld_lst, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.legend()
    plt.show()

    # Plot training and validation log_likelihood as a function of iterations
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    plt.plot(iter_lst, train_lld_lst, label="Train")
    plt.plot(iter_lst, val_lld_lst, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.show()

    # Report the final validation and test accuracy
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation Accuracy: ", val_acc)
    print("Test Accuracy: ", test_acc)

    # Select five questions (betas), plot the possibility of correctness
    # as a function of students (thetas)
    theta_lst = np.arange(-5, 5, 0.01)
    beta_lst = [3, 29, 107, 593, 1771]
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})
    for j in beta_lst:
        p = sigmoid(theta_lst - beta[j])
        plt.plot(theta_lst, p, label="Question {}".format(j))
    plt.xlabel("Students(theta)")
    plt.ylabel("Possibility of Correctness")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
