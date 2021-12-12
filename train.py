import time
import argparse
import os
import sys
if sys.version_info >= (3, 0):
        import _pickle as cPickle
else:
        import cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_loader import load_data 
from parameters import DATASET, TRAINING, HYPERPARAMS

def train(epochs=HYPERPARAMS.epochs, random_state=HYPERPARAMS.random_state, 
          kernel=HYPERPARAMS.kernel, decision_function=HYPERPARAMS.decision_function, gamma=HYPERPARAMS.gamma, train_model=True):

        print( "加载数据集 " + DATASET.name + "...")
        if train_model:
                data, validation = load_data(validation=True)
        else:
                data, validation, test = load_data(validation=True, test=True)
        
        if train_model:
            # 训练阶段
            print( "建立模型...")
            model = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function, gamma=gamma)

            print( "开始训练...")
            print( "--")
            print( "内核: {}".format(kernel))
            print( "决策函数: {} ".format(decision_function))
            print( "max epochs: {} ".format(epochs))
            print( "gamma: {} ".format(gamma))
            print( "--")
            print( "训练样本: {}".format(len(data['Y'])))
            print( "检验样本: {}".format(len(validation['Y'])))
            print( "--")
            start_time = time.time()
            model.fit(data['X'], data['Y'])
            training_time = time.time() - start_time
            print( "训练时长 = {0:.1f} 秒".format(training_time))

            if TRAINING.save_model:
                print( "保存模型...")
                with open(TRAINING.save_model_path, 'wb') as f:
                        cPickle.dump(model, f)

            print( "评估中...")
            validation_accuracy = evaluate(model, validation['X'], validation['Y'])
            print( "  - 评估准确性 = {0:.1f}".format(validation_accuracy*100))
            return validation_accuracy
        else:
            # 测试阶段:负载保存模型和测试数据集评估
            print( "开始评估...")
            print( "加载预训练模型...")
            if os.path.isfile(TRAINING.save_model_path):
                with open(TRAINING.save_model_path, 'rb') as f:
                        model = cPickle.load(f)
            else:
                print( "Error: file '{}' not found".format(TRAINING.save_model_path))
                exit()

            print( "--")
            print( "检验样本: {}".format(len(validation['Y'])))
            print( "测试样本: {}".format(len(test['Y'])))
            print( "--")
            print( "评估中...")
            start_time = time.time()
            validation_accuracy = evaluate(model, validation['X'],  validation['Y'])
            print( "  - 评估准确性 = {0:.1f}".format(validation_accuracy*100))
            test_accuracy = evaluate(model, test['X'], test['Y'])
            print( "  - 测试准确性 = {0:.1f}".format(test_accuracy*100))
            print( "  - 评估市时长 = {0:.1f} 秒".format(time.time() - start_time))
            return test_accuracy

def evaluate(model, X, Y):
        predicted_Y = model.predict(X)
        accuracy = accuracy_score(Y, predicted_Y)
        return accuracy

# 解析arg，看看我们是否需要现在启动训练
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="如果为 'yes', 从命令行启动训练")
parser.add_argument("-e", "--evaluate", default="no", help="如果为 'yes', 启动测试数据集的评估")
parser.add_argument("-m", "--max_evals", help="超参数搜索期间的最大计算次数")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
        train(train_model=False)