#!/usr/bin/env python
import numpy as np
from utils import *


if __name__ == '__main__':
    dataset = read_file("activity_recognition_dataset.csv")
    train,test = data_preprocess(dataset, 0.8)

    print(train.shape)
    print(test.shape)
