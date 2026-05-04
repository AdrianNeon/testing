import math

# 2.1
def z_score_scaler(features):
	if not features or len(features) == 0:
		return [], [], []

	n_features = len(features[0])
	scaled_features = []
	means = []
	stds = []

	for i in range(n_features):
		values = [sample[i] for sample in features]
		mean_val = sum(values) / len(values)
		means.append(mean_val)

		variance = sum((v - mean_val) ** 2 for v in values) / len(values)
		std_val = math.sqrt(variance)
		stds.append(std_val)

	for sample in features:
		scaled_sample = []
		for i in range(n_features):
			if stds[i] == 0:
				scaled_val = 0.0  # 标准差为0时，标准化后统一为0
			else:
				scaled_val = (sample[i] - means[i]) / stds[i]
			scaled_sample.append(scaled_val)
		scaled_features.append(scaled_sample)

	return scaled_features, means, stds


if __name__ == "__main__":
	# 构建测试数据（基于模块一的数据集）
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

	features = [sample[0] for sample in dataset]

	print("原始特征矩阵:")
	for f in features:
		print(f)

	n_features = len(features[0])
	print("\n标准化前:")
	for i in range(n_features):
		values = [f[i] for f in features]
		mean_val = sum(values) / len(values)
		variance = sum((v - mean_val) ** 2 for v in values) / len(values)
		std_val = math.sqrt(variance)
		print(f"特征{i}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")

	scaled_features, means, stds = z_score_scaler(features)

	print("\n标准化后特征矩阵:")
	for f in scaled_features:
		print([round(x, 4) for x in f])

	print("\n标准化后:")
	for i in range(n_features):
		values = [f[i] for f in scaled_features]
		mean_val = sum(values) / len(values)
		variance = sum((v - mean_val) ** 2 for v in values) / len(values)
		std_val = math.sqrt(variance)
		print(f"特征{i}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")


# 2.2
def fill_missing_with_median(features):
	if not features or len(features) == 0:
		return [], []

	n_features = len(features[0])
	filled_features = [sample.copy() for sample in features]
	fill_values = []

	for i in range(n_features):
		valid_values = [sample[i] for sample in features if sample[i] is not None]

		if valid_values:
			sorted_vals = sorted(valid_values)
			n = len(sorted_vals)
			if n % 2 == 1:
				median_val = sorted_vals[n // 2]
			else:
				median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
		else:
			median_val = 0.0

		fill_values.append(median_val)

		for sample in filled_features:
			if sample[i] is None:
				sample[i] = median_val

	return filled_features, fill_values


if __name__ == "__main__":
	features_with_nan = [
		[5.1, 3.5, None],
		[4.9, None, 1.4],
		[None, 3.2, 1.3],
		[4.6, 3.1, 1.5],
		[5.0, 3.6, None],
		[None, 3.2, 4.7],
		[6.4, None, 4.5],
		[6.9, 3.1, None],
		[5.5, 2.3, 4.0],
		[6.5, 2.8, 4.6]
	]

	print("\n原始特征矩阵（含缺失值None）:")
	for f in features_with_nan:
		print(f)

	filled_features, fill_vals = fill_missing_with_median(features_with_nan)

	print(f"\n填充值（中位数）: {fill_vals}")
	print("\n填充后特征矩阵:")
	for f in filled_features:
		print([round(x, 4) if isinstance(x, float) else x for x in f])


# 2.3
def label_encoder(categories):
	if not categories:
		return [], {}

	unique_categories = list(dict.fromkeys(categories))  # 保持顺序的去重

	mapping = {cat: idx for idx, cat in enumerate(unique_categories)}

	encoded = [mapping[cat] for cat in categories]

	return encoded, mapping


if __name__ == "__main__":
	weather = ["Sunny", "Rainy", "Sunny", "Cloudy", "Rainy", "Cloudy", "Sunny"]

	print(f"\n原始离散特征: {weather}")

	encoded, mapping = label_encoder(weather)

	print(f"编码映射: {mapping}")
	print(f"编码后数值列表: {encoded}")

	decoded = [list(mapping.keys())[list(mapping.values()).index(v)] for v in encoded]
	print(f"解码验证: {decoded}")