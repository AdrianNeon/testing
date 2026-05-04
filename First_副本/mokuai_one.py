import random

#模块一
# 2.1
def printDataSet(data):
    length = len(data)
    values = [sample[0] for sample in data]
    labels = [sample[1] for sample in data]
    for i in range(length):
        print(f'特征：{values[i]}')
        print(f'标签：{labels[i]}')

# 2.2
def printChar(data):
    length = len(data)
    values = [sample[0] for sample in data]
    labels = [sample[1] for sample in data]
    pos_labels = sum(1 for label in labels if label == 1)
    neg_labels = length - pos_labels

    print(f'总数： {length}')
    print(f'特征维度： {len(values[0])}')
    print(f'正： {pos_labels}')
    print(f'负： {neg_labels}')

# 2.3
def difference(data):
    length = len(data)
    values = [sample[0] for sample in data]
    dimension = len(values[0])

    for i in range(length):
        avg = sum(values[i]) / dimension
        diff = 0
        for x in values[i]:
            diff += (x - avg) ** 2

        print(diff / dimension)

# 2.4
def getLargest(data):
    length = len(data)
    values = [sample[0] for sample in data]
    labels = [sample[1] for sample in data]

    for i in range(length):
        if values[i][0] > 5.0:
            print(f'特征：{values[i]}')
            print(f'标签：{labels[i]}')

# 3.0
def randomize(data):
    length = len(data)
    random.shuffle(data)
    values = [sample[0] for sample in data]
    labels = [sample[1] for sample in data]
    training = data[0: 7]
    testing = data[7: 10]
    print("训练：")
    print(training)
    print(f"Total: {len(training)}")
    print("测试：")
    print(testing)
    print(f"Total: {len(testing)}")

# Extra 1
def dict_data(dataset):
    dict_dataset = []

    for i, sample in enumerate(dataset):
        sample = {
            "ID": i,
            "val": sample[0],
            "label": sample[1]
        }
        dict_dataset.append(sample)

    for i in range(10):
        print(dict_dataset[i])

# 1
dataset = [
          [[5.1, 3.5, 4.2], 0],
          [[4.9, 3.0, 2.1], 0],
          [[4.7, 3.2, 6.4], 0],
          [[4.6, 3.1, 2.2], 0],
          [[7.0, 3.2, 1.3], 0],
          [[6.4, 3.2, 7.5], 1],
          [[6.9, 3.1, 1.7], 1],
          [[5.5, 2.3, 6.3], 1],
          [[2.3, 5.1, 4.6], 1],
          [[6.1, 2.7, 3.1], 1],
        ]

# 2.1
print("2.1")
printDataSet(dataset)
print()

# 2.2
print("2.2")
printChar(dataset)
print()


# 2.3
print("2.3")
difference(dataset)
print()


# 2.4
print("2.4")
getLargest(dataset)
print()


# 3.0
print("3.0")
randomize(dataset)
print()


# Extra 1
dict_data(dataset)
print()