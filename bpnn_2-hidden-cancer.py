# -*- coding: utf-8 -*- 
import math
import random
import numpy as np
import matplotlib.pyplot as plt  
random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmod_derivate(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0 	# 输入层个数
        self.hidden_n = 0 	# 隐藏层1个数
        self.hidden2_n = 0 	# 隐藏层2个数
        self.output_n = 0 	# 输出层个数
        self.input_cells = []
        self.hidden_cells = []
        self.hidden2_cells = []
        self.output_cells = []
        self.input_weights = [] 	# 输入层权值
        self.hidden_weights = [] 	# 隐藏层权值
        self.output_weights = [] 	# 输出层权值
        self.input_correction = []	# 输入层偏差矫正
        self.hidden_correction = []	# 隐藏层偏差矫正
        self.output_correction = []	# 输出层偏差矫正

    def setup(self, ni, nh, nh2, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.hidden2_n = nh2
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.hidden2_cells = [1.0] * self.hidden2_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.hidden_weights = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_weights = make_matrix(self.hidden2_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for hh in range(self.hidden2_n):
                self.hidden_weights[i][hh] = rand(-0.2, 0.2)
        for hh in range(self.hidden2_n):
            for o in range(self.output_n):
                self.output_weights[hh][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.hidden_correction = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_correction = make_matrix(self.hidden2_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
		# activate hidden2 layer
        for j in range(self.hidden2_n):# 隐藏层
            total = 0.0
            for i in range(self.hidden_n):
                total += self.hidden_cells[i] * self.hidden_weights[i][j]
            self.hidden2_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden2_n):
                total += self.hidden2_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label - self.output_cells[o]
            output_deltas[o] = sigmod_derivate(self.output_cells[o]) * error
        # get hidden2 layer error 隐藏层2
        hidden2_deltas = [0.0] * self.hidden2_n
        for hh in range(self.hidden2_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[hh][o]
            hidden2_deltas[hh] = sigmod_derivate(self.hidden2_cells[hh]) * error
        # get hidden layer error 隐藏层1
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.hidden2_n):
                error += hidden2_deltas[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = sigmod_derivate(self.hidden_cells[h]) * error
        # update output weights 更新输出层权值
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update hidden weights 更新隐藏层权值
        for h in range(self.hidden_n):
            for o in range(self.hidden2_n):
                change = hidden2_deltas[o] * self.hidden_cells[h]
                self.hidden_weights[h][o] += learn * change + correct * self.hidden_correction[h][o]
                self.hidden_correction[h][o] = change
        # update input weights  更新输入层权值
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        # for o in range(len(label)):
        error += 0.5 * (label - self.output_cells[0]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        error_results = []
        nums = []
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            if j%50==0 :
                print j , error
                error_results.append(error)
                nums.append(j)
        #X轴的文字  
        plt.xlabel("Num")  
        #Y轴的文字  
        plt.ylabel("Error")  
        #图表的标题  
        plt.title("BPNN") 
        plt.xlim(0.0, 4000.0)# set axis limits
        plt.ylim(0.0, 20.)
        
        plt.plot(nums, error_results)# use pylab to plot x and y
        plt.show()# show the plot on the screen

    def test(self):
        dataset = np.loadtxt('cancer.csv', delimiter=",")
        cases = dataset[:,1:10]
        labels = dataset[:,10]
        self.setup(8, 17,5, 1)
        self.train(cases[:400], labels, 4000, 0.05, 0.1)
        count = [0,0]
        i=0
        for case in cases[400:]:
#             print(self.predict(case)[0])
            if abs(self.predict(case)[0]-labels[400+i])<0.1:
                count[0]=count[0]+1
            else: 
                count[1]=count[1]+1
            i=i+1
        print 'result =',count
        print 'error =',float(count[1])/(count[0]+count[1])


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
