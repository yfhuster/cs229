import numpy as np
import matplotlib.pyplot as plt
import svm

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)


'''
    注释掉的代码使用Bernoulli event model,特征向量的每一位编码一个固定的词汇,1代表出现,0代表不出现
    未注释的使用Multinomial event model
'''
def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    '''
    state_spam, state_news = [], []
    spam = (category == 1)
    news = (category == 0)
    num_spam = sum(spam)
    num_news = sum(news)
    spam_ratio = num_spam / matrix.shape[0]
    news_ratio = num_news / matrix.shape[0]
    for j in range(N):
        word_spam = (1 + sum(matrix[spam, j] == 1)) / (num_spam + 2)
        word_news = (1 + sum(matrix[news, j] == 1)) / (num_news + 2)
        state_spam.append(word_spam)
        state_news.append(word_news)
    state['pro_sp'] = spam_ratio
    state['pro_ne'] = news_ratio
    state['pro_spwords'] = state_spam
    state['pro_newords'] = state_news
    '''

    spam = (category == 1)
    news = (category == 0)
    length_spam = np.sum(matrix[spam, :], axis=1)
    length_news = np.sum(matrix[news, :], axis=1)
    spam_ratio = np.sum(spam) / matrix.shape[0]
    news_ratio = np.sum(news) / matrix.shape[0]
    state['pro_sp'] = spam_ratio
    state['pro_ne'] = news_ratio
    state['pro_spwords'] = (1 + np.sum(matrix[spam, :], axis=0)) / (np.sum(length_spam) + N)
    state['pro_newords'] = (1 + np.sum(matrix[news, :], axis=0)) / (np.sum(length_news) + N)
    # print(state)
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    '''
    words_spam = np.array(state['pro_spwords'])
    pro_spam_test = matrix.dot(np.log(words_spam)) + (1 - matrix).dot(np.log(1 - words_spam))
    pro_spam_test += np.log(state['pro_sp'])

    words_news = np.array(state['pro_newords'])
    pro_news_test = matrix.dot(np.log(words_news)) + (1 - matrix).dot(np.log(1 - words_news))
    pro_news_test += np.log(state['pro_ne'])

    output[pro_spam_test > pro_news_test] = 1
    '''

    log_spam = np.log(state['pro_spwords'])
    log_news = np.log(state['pro_newords'])
    pro_spam_test = np.dot(matrix, log_spam) + np.log(state['pro_sp'])
    pro_news_test = np.dot(matrix, log_news) + np.log(state['pro_ne'])

    output[pro_spam_test > pro_news_test] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print ('Error: %1.4f' % error)
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    print('train: {}  test: {}'.format(trainMatrix.shape, testMatrix.shape))

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    indicate = np.argsort(state['pro_spwords'] / state['pro_newords'])[-5:]
    print(np.array(tokenlist)[indicate])

    train_list = ['MATRIX.TRAIN.50', 'MATRIX.TRAIN.100', 'MATRIX.TRAIN.200', 'MATRIX.TRAIN.400',
                  'MATRIX.TRAIN.800', 'MATRIX.TRAIN.1400']
    train_sizes = np.array([50, 100, 200, 400, 800, 1400])
    errors = np.zeros(train_sizes.shape)
    errors_svm = np.zeros(train_sizes.shape)
    for i, train_file in enumerate(train_list):
        trainMatrix, tokenlist, trainCategory = readMatrix(train_file)
        state = nb_train(trainMatrix, trainCategory)
        output = nb_test(testMatrix, state)
        errors[i] = evaluate(output, testCategory)

    errors_svm = svm.main()

    plt.plot(train_sizes, errors*100, label='Naive Bayes')
    plt.plot(train_sizes, errors_svm*100, label='SVM')
    plt.legend()
    plt.xlabel('Training Size')
    plt.ylabel('Test Errors(%)')
    plt.show()
    return

if __name__ == '__main__':
    main()
