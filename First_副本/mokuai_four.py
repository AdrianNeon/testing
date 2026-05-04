def calc_classification_metrics(y_true, y_pred, pos_label=1):
	TP = TN = FP = FN = 0

	for true, pred in zip(y_true, y_pred):
		if true == pos_label and pred == pos_label:
			TP += 1
		elif true != pos_label and pred != pos_label:
			TN += 1
		elif true != pos_label and pred == pos_label:
			FP += 1
		elif true == pos_label and pred != pos_label:
			FN += 1

	total = TP + TN + FP + FN
	accuracy = (TP + TN) / total if total != 0 else 0.0
	precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
	recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

	return {
		"混淆矩阵": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
		"准确率": round(accuracy, 4),
		"精确率": round(precision, 4),
		"召回率": round(recall, 4),
		"F1值": round(f1, 4)
	}


if __name__ == "__main__":
	test_true_labels = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

	predictions_k1 = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
	predictions_k3 = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
	predictions_k5 = [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]

	print("不同k值下的模型评估指标")

	for k, preds in [("k=1", predictions_k1), ("k=3", predictions_k3), ("k=5", predictions_k5)]:
		print(f"\n{k}:")
		metrics = calc_classification_metrics(test_true_labels, preds)
		for key, value in metrics.items():
			print(f"  {key}: {value}")