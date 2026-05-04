from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']

class PerceptronSGD:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for epoch in range(self.max_iter):
            loss = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    loss += 1
            self.losses.append(loss)
            if loss == 0:
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)


class PerceptronBGD:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for epoch in range(self.max_iter):
            # 批量梯度：累积所有误分类样本的梯度
            grad_w = np.zeros(n_features)
            grad_b = 0
            loss = 0

            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    grad_w += yi * xi
                    grad_b += yi
                    loss += 1

            self.w += self.lr * grad_w
            self.b += self.lr * grad_b
            self.losses.append(loss)

            if loss == 0:
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)


class PerceptronMiniBatch:
    def __init__(self, lr=0.01, batch_size=32, max_iter=1000):
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for epoch in range(self.max_iter):
            loss = 0
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                grad_w = np.zeros(n_features)
                grad_b = 0
                batch_loss = 0

                for xi, yi in zip(X_batch, y_batch):
                    if yi * (np.dot(self.w, xi) + self.b) <= 0:
                        grad_w += yi * xi
                        grad_b += yi
                        batch_loss += 1

                if batch_loss > 0:
                    self.w += self.lr * grad_w
                    self.b += self.lr * grad_b
                loss += batch_loss

            self.losses.append(loss)
            if loss == 0:
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)


class PerceptronDual:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
        self.losses = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.losses = []

        gram_matrix = np.dot(X, X.T)

        for epoch in range(self.max_iter):
            loss = 0
            for i in range(n_samples):
                decision = self.b
                for j in range(n_samples):
                    decision += self.alpha[j] * self.y[j] * gram_matrix[j, i]

                if self.y[i] * decision <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr * self.y[i]
                    loss += 1

            self.losses.append(loss)
            if loss == 0:
                break
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            decision = self.b
            for i in range(len(self.alpha)):
                decision += self.alpha[i] * self.y[i] * np.dot(self.X[i], x)
            predictions.append(1 if decision >= 0 else -1)
        return np.array(predictions)


class PerceptronAdaptiveLR:
    def __init__(self, initial_lr=0.1, decay_rate=0.01, max_iter=1000):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.losses = []
        self.lr_history = []

    def _get_lr(self, t):
        return self.initial_lr / (1 + self.decay_rate * t)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []
        self.lr_history = []

        for epoch in range(self.max_iter):
            current_lr = self._get_lr(epoch)
            self.lr_history.append(current_lr)
            loss = 0

            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += current_lr * yi * xi
                    self.b += current_lr * yi
                    loss += 1

            self.losses.append(loss)
            if loss == 0:
                break
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, -1)


print("=" * 60)
print("感知机模型实验")
print("=" * 60)

print("\n【步骤1】数据准备...")
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_clusters_per_class=1, random_state=42)
y = np.where(y == 0, -1, 1)  # 标签映射为 -1, 1

print("【步骤2】数据预处理（标准化 + 划分训练/测试集）...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"   训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

print("\n" + "=" * 60)
print("【实验1】不同学习率对收敛速度的影响（SGD）")
print("=" * 60)

learning_rates = [0.001, 0.01, 0.1]
models_sgd = {}
colors = ['red', 'green', 'blue']

for lr, color in zip(learning_rates, colors):
    perceptron = PerceptronSGD(lr=lr, max_iter=100)
    perceptron.fit(X_train, y_train)
    models_sgd[lr] = perceptron

    y_pred = perceptron.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    converge_epoch = None
    for i, loss in enumerate(perceptron.losses):
        if loss == 0:
            converge_epoch = i
            break

    print(f"\n学习率 = {lr}")
    print(f"  最终迭代轮数: {len(perceptron.losses)}")
    print(f"  最终误分类数: {perceptron.losses[-1]}")
    print(f"  收敛轮数: {converge_epoch if converge_epoch is not None else '未收敛'}")
    print(f"  测试准确率: {accuracy:.4f}")

fig = plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for lr, color in zip(learning_rates, colors):
    plt.plot(range(len(models_sgd[lr].losses)), models_sgd[lr].losses,
             color=color, label=f'lr = {lr}', linewidth=2)
plt.xlabel('迭代轮数 (Epoch)')
plt.ylabel('误分类样本数')
plt.title('训练损失曲线对比\n(不同学习率)')
plt.legend()
plt.grid(True, alpha=0.3)

for i, (lr, color) in enumerate(zip(learning_rates, colors), start=2):
    plt.subplot(2, 3, i)
    plt.plot(range(len(models_sgd[lr].losses)), models_sgd[lr].losses,
             color=color, linewidth=2)
    plt.xlabel('迭代轮数 (Epoch)')
    plt.ylabel('误分类样本数')
    plt.title(f'学习率 = {lr}\n(最终损失: {models_sgd[lr].losses[-1]})')
    plt.grid(True, alpha=0.3)


def plot_decision_boundary(model, X, y, ax, title):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    ax.set_title(title)


for i, lr in enumerate(learning_rates):
    ax = plt.subplot(2, 3, i + 4)
    acc = np.mean(models_sgd[lr].predict(X_test) == y_test)
    plot_decision_boundary(models_sgd[lr], X_train, y_train, ax,
                          f'决策边界 (lr={lr})\n测试准确率: {acc:.3f}')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("【实验2】比较SGD、BGD、Mini-batch GD")
print("=" * 60)

best_lr = 0.1

print(f"\n使用学习率 = {best_lr}")

sgd_model = PerceptronSGD(lr=best_lr, max_iter=100)
sgd_model.fit(X_train, y_train)
sgd_acc = np.mean(sgd_model.predict(X_test) == y_test)
print(f"SGD (随机梯度下降): 测试准确率 = {sgd_acc:.4f}, 收敛轮数 = {len(sgd_model.losses)}")

bgd_model = PerceptronBGD(lr=best_lr, max_iter=100)
bgd_model.fit(X_train, y_train)
bgd_acc = np.mean(bgd_model.predict(X_test) == y_test)
print(f"BGD (批量梯度下降): 测试准确率 = {bgd_acc:.4f}, 收敛轮数 = {len(bgd_model.losses)}")

minibatch_model = PerceptronMiniBatch(lr=best_lr, batch_size=32, max_iter=100)
minibatch_model.fit(X_train, y_train)
minibatch_acc = np.mean(minibatch_model.predict(X_test) == y_test)
print(f"Mini-batch GD (小批量梯度下降, batch_size=32): 测试准确率 = {minibatch_acc:.4f}, 收敛轮数 = {len(minibatch_model.losses)}")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(range(len(sgd_model.losses)), sgd_model.losses, label='SGD', linewidth=2, color='blue')
ax.plot(range(len(bgd_model.losses)), bgd_model.losses, label='BGD', linewidth=2, color='red')
ax.plot(range(len(minibatch_model.losses)), minibatch_model.losses, label='Mini-batch GD', linewidth=2, color='green')
ax.set_xlabel('迭代轮数 (Epoch)')
ax.set_ylabel('误分类样本数')
ax.set_title('不同梯度下降方法的收敛对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 60)
print("【实验3】对偶形式感知机")
print("=" * 60)

dual_model = PerceptronDual(lr=0.1, max_iter=100)
dual_model.fit(X_train, y_train)
dual_acc = np.mean(dual_model.predict(X_test) == y_test)

print(f"对偶形式感知机: 测试准确率 = {dual_acc:.4f}")
print(f"Gram矩阵大小: {len(X_train)} x {len(X_train)} = {len(X_train)**2} 个元素")
print("Gram矩阵作用: 预先计算所有样本对的内积，避免重复计算，加速训练")

print("\n" + "=" * 60)
print("【实验4】自适应学习率策略")
print("=" * 60)

adaptive_model = PerceptronAdaptiveLR(initial_lr=0.1, decay_rate=0.01, max_iter=100)
adaptive_model.fit(X_train, y_train)
adaptive_acc = np.mean(adaptive_model.predict(X_test) == y_test)

print(f"自适应学习率感知机: 测试准确率 = {adaptive_acc:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(len(adaptive_model.lr_history)), adaptive_model.lr_history, 'b-', linewidth=2)
ax1.set_xlabel('迭代轮数 (Epoch)')
ax1.set_ylabel('学习率')
ax1.set_title('自适应学习率衰减曲线')
ax1.grid(True, alpha=0.3)

ax2.plot(range(len(adaptive_model.losses)), adaptive_model.losses, 'r-', linewidth=2)
ax2.set_xlabel('迭代轮数 (Epoch)')
ax2.set_ylabel('误分类样本数')
ax2.set_title('自适应学习率 - 损失曲线')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("【实验结果总结】")
print("=" * 60)

print("\n1. 学习率影响分析:")
print("   - 学习率过小(0.001): 收敛速度慢，可能需要更多迭代")
print("   - 学习率适中(0.01, 0.1): 收敛速度快，效果好")
print("   - 学习率过大: 可能导致震荡，不易收敛")

print("\n2. 梯度下降方法对比:")
print("   - SGD: 更新频繁，收敛快但可能有波动")
print("   - BGD: 更新稳定，但每次迭代计算量大")
print("   - Mini-batch: 折中方案，兼顾效率与稳定性")

print("\n3. 对偶形式优势:")
print("   - 利用Gram矩阵缓存内积结果")
print("   - 训练时间复杂度从O(T*n*d)降为O(T*n^2)")
print("   - 但Gram矩阵空间复杂度为O(n^2)，大数据集内存开销大")

print("\n4. 自适应学习率:")
print("   - 初期学习率大，快速收敛")
print("   - 后期学习率小，精细调整")
print("   - 公式: α_t = α_0 / (1 + decay_rate * t)")