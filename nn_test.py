from models import nn
from utils import read_file, data_preprocess, print_accuracy_stats

dataset = read_file("activity_recognition_dataset.csv")
train,test = data_preprocess(dataset, 0.8)
ts_y = test[:, -1:]
x = nn.NeuralNet();
x.train(train)
pred = x.test(test)
print_accuracy_stats(pred, ts_y)