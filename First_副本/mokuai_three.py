import math
import random
from mokuai_two import z_score_scaler

def calc_euclidean_distance(x1, x2):
	"""计算两个样本之间的欧式距离"""
	if len(x1) != len(x2):
		raise ValueError("两个样本的特征维度必须一致")
	squared_sum = 0.0
	for a, b in zip(x1, x2):
		squared_sum += (a - b) ** 2
	return math.sqrt(squared_sum)


def get_k_neighbors(train_set, test_sample, k):
	distances = []
	for idx, (features, label) in enumerate(train_set):
		dist = calc_euclidean_distance(test_sample, features)
		distances.append((dist, label, idx))

	distances.sort(key=lambda x: x[0])

	k_neighbors = distances[:k]
	return [neighbor[1] for neighbor in k_neighbors]


def vote_labels(neighbor_labels):
	label_count = {}
	for label in neighbor_labels:
		label_count[label] = label_count.get(label, 0) + 1

	max_count = max(label_count.values())
	candidates = [label for label, count in label_count.items() if count == max_count]
	return min(candidates)


def knn_classifier(train_set, test_set, k):
	predictions = []
	for test_sample, _ in test_set:
		neighbor_labels = get_k_neighbors(train_set, test_sample, k)
		predicted_label = vote_labels(neighbor_labels)
		predictions.append(predicted_label)
	return predictions


def calc_manhattan_distance(x1, x2):
	if len(x1) != len(x2):
		raise ValueError("两个样本的特征维度必须一致")
	distance = 0.0
	for a, b in zip(x1, x2):
		distance += abs(a - b)
	return distance


def knn_classifier_manhattan(train_set, test_set, k):
	predictions = []
	for test_sample, _ in test_set:
		distances = []
		for idx, (features, label) in enumerate(train_set):
			dist = calc_manhattan_distance(test_sample, features)
			distances.append((dist, label, idx))
		distances.sort(key=lambda x: x[0])
		k_neighbors = distances[:k]
		neighbor_labels = [neighbor[1] for neighbor in k_neighbors]

		label_count = {}
		for label in neighbor_labels:
			label_count[label] = label_count.get(label, 0) + 1
		max_count = max(label_count.values())
		candidates = [label for label, count in label_count.items() if count == max_count]
		predicted_label = min(candidates)
		predictions.append(predicted_label)
	return predictions


if __name__ == "__main__":
	dataset = [
		[[5.1, 3.5, 1.4], 0],
		[[4.9, 3.0, 1.4], 0],
		[[4.7, 3.2, 1.3], 0],
		[[4.6, 3.1, 1.5], 0],
		[[5.0, 3.6, 1.4], 0],
		[[7.0, 3.2, 4.7], 1],
		[[6.4, 3.2, 4.5], 1],
		[[6.9, 3.1, 4.9], 1],
		[[5.5, 2.3, 4.0], 1],
		[[6.5, 2.8, 4.6], 1]
	]

	random.seed(42)
	shuffled_dataset = dataset.copy()
	random.shuffle(shuffled_dataset)

	split_idx = int(len(shuffled_dataset) * 0.7)
	train_set = shuffled_dataset[:split_idx]
	test_set = shuffled_dataset[split_idx:]

	print(f"训练集样本数: {len(train_set)}")
	print(f"测试集样本数: {len(test_set)}")

	train_features = [sample[0] for sample in train_set]
	test_features = [sample[0] for sample in test_set]

	scaled_train_features, means, stds = z_score_scaler(train_features)

	scaled_test_features = []
	for sample in test_features:
		scaled_sample = []
		for i in range(len(sample)):
			if stds[i] == 0:
				scaled_val = 0.0
			else:
				scaled_val = (sample[i] - means[i]) / stds[i]
			scaled_sample.append(scaled_val)
		scaled_test_features.append(scaled_sample)

	scaled_train_set = [[scaled_train_features[i], train_set[i][1]] for i in range(len(train_set))]
	scaled_test_set = [[scaled_test_features[i], test_set[i][1]] for i in range(len(test_set))]

	test_true_labels = [label for _, label in test_set]

	for k in [1, 3, 5]:
		print(f"\n{'=' * 40}")
		print(f"k = {k} 时的预测结果:")
		predictions = knn_classifier(scaled_train_set, scaled_test_set, k)
		print(f"真实标签: {test_true_labels}")
		print(f"预测标签: {predictions}")

		correct = sum(1 for true, pred in zip(test_true_labels, predictions) if true == pred)
		accuracy = correct / len(test_true_labels)
		print(f"准确率: {accuracy:.4f}")

		print(f"\n使用曼哈顿距离 (k={k}):")
		predictions_manhattan = knn_classifier_manhattan(scaled_train_set, scaled_test_set, k)
		print(f"预测标签: {predictions_manhattan}")
		correct_m = sum(1 for true, pred in zip(test_true_labels, predictions_manhattan) if true == pred)
		accuracy_m = correct_m / len(test_true_labels)
		print(f"准确率: {accuracy_m:.4f}")