import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import numpy as np
import platform

def setup_chinese_font():
	system = platform.system()

	if system == 'Windows':
		fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
	elif system == 'Darwin':
		fonts = ['Arial Unicode MS', 'Heiti SC', 'STHeiti', 'PingFang SC']

	for font in fonts:
		try:
			plt.rcParams['font.sans-serif'] = [font]
			break
		except:
			continue

	plt.rcParams['axes.unicode_minus'] = False

	return plt.rcParams['font.sans-serif'][0]

current_font = setup_chinese_font()

attr_values = [
	['Sunny', 'Rainly'],
	['Warm', 'Cold'],
	['Normal', 'High'],
	['Strong', 'Light'],
	['Warm', 'Cool'],
	['Same', 'Change']
]

attr_names = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']

def agree(h, e, elabel):
	covers = True
	for i in range(len(h)):
		if h[i] not in ('?', e[i]):
			covers = False
			break

	if elabel:
		return covers
	else:
		return not covers


def is_general_than(h1, h2):
	for i in range(len(h1)):
		if h1[i] != '?' and h1[i] != h2[i]:
			return False
	return True


def is_special_than(h1, h2):
	return is_general_than(h2, h1)


def min_generalize(h, e):
	new_h = list(h)
	for i in range(len(h)):
		if h[i] == '$':
			new_h[i] = e[i]
		elif h[i] != e[i] and h[i] != '?':
			new_h[i] = '?'
	return new_h


def min_specialize(h):
	specializations = []
	for i in range(len(h)):
		if h[i] == '?':
			for val in attr_values[i]:
				new_h = list(h)
				new_h[i] = val
				specializations.append(new_h)
	return specializations


def generate_SG(examples):
	num_attrs = len(attr_values)

	S = [['$'] * num_attrs]
	G = [['?'] * num_attrs]

	for idx, (e, label) in enumerate(examples):
		print(f"\n--- 处理样例 {idx + 1}: {e}, 标签={'正例' if label else '负例'} ---")

		if label:
			G = [g for g in G if agree(g, e, True)]

			new_S = []
			for s in S:
				if not agree(s, e, True):
					new_s = min_generalize(s, e)
					new_S.append(new_s)
				else:
					new_S.append(s)
			S = new_S

			S = [s for s in S if any(is_special_than(s, g) for g in G)]

		else:
			S = [s for s in S if agree(s, e, False)]

			new_G = []
			for g in G:
				if agree(g, e, True):
					specials = min_specialize(g)
					for spec in specials:
						if not agree(spec, e, True):
							new_G.append(spec)
				else:
					new_G.append(g)

			unique_G = []
			for g in new_G:
				if g not in unique_G:
					unique_G.append(g)
			G = unique_G

			G = [g for g in G if any(is_general_than(g, s) for s in S)]

		print(f"  S: {S}")
		print(f"  G: {G}")

	return S, G


def format_hypothesis_short(h):
	"""格式化假设为简短字符串"""
	if not h:
		return "None"
	# 检查是否为初始状态
	if all(x == '$' for x in h):
		return "初始 (全$)"
	if all(x == '?' for x in h):
		return "全通配"

	parts = []
	for i, val in enumerate(h):
		if val != '?':
			parts.append(f"{attr_names[i][:3]}={val}")

	if not parts:
		return "全通配"
	if len(parts) > 2:
		return ", ".join(parts[:2]) + "..."
	return ", ".join(parts)


def draw_hypothesis_tree(S, G, title="变型空间可视化"):
	fig, ax = plt.subplots(figsize=(14, 8))
	ax.set_xlim(0, 12)
	ax.set_ylim(0, 10)
	ax.axis('off')
	ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

	max_hypotheses = max(len(S), len(G), 1)
	spacing = min(2.5, 10 / max_hypotheses)
	start_x = (12 - (max_hypotheses - 1) * spacing) / 2

	y_s = 2.5
	s_bg = patches.FancyBboxPatch((0.5, y_s - 0.8), 11, 1.6,
								  boxstyle="round,pad=0.1",
								  facecolor='#E6F0FA', edgecolor='#4472C4', alpha=0.3)
	ax.add_patch(s_bg)
	ax.text(6, y_s + 0.6, "S 边界 (最特殊假设)", ha='center', fontsize=12,
			fontweight='bold', color='#4472C4')

	if S and S[0] and S[0][0] != '$':
		for i, s in enumerate(S):
			x = start_x + i * spacing
			x = max(1, min(11, x))

			rect = patches.FancyBboxPatch((x - 1.2, y_s - 0.5), 2.4, 0.9,
										  boxstyle="round,pad=0.1",
										  facecolor='#4472C4', edgecolor='#2E5A9E', alpha=0.9)
			ax.add_patch(rect)

			# 显示假设文本
			text = format_hypothesis_short(s)
			ax.text(x, y_s, text, ha='center', va='center',
					fontsize=9, color='white', fontweight='bold')
	else:
		ax.text(6, y_s, "初始状态或空集", ha='center', va='center',
				fontsize=11, color='gray', style='italic')

	y_g = 7.0
	g_bg = patches.FancyBboxPatch((0.5, y_g - 0.8), 11, 1.6,
								  boxstyle="round,pad=0.1",
								  facecolor='#E8F5E9', edgecolor='#70AD47', alpha=0.3)
	ax.add_patch(g_bg)
	ax.text(6, y_g + 0.6, "G 边界 (最一般假设)", ha='center', fontsize=12,
			fontweight='bold', color='#70AD47')

	if G and G[0] and G[0][0] != '?':
		for i, g in enumerate(G):
			x = start_x + i * spacing
			x = max(1, min(11, x))

			rect = patches.FancyBboxPatch((x - 1.2, y_g - 0.5), 2.4, 0.9,
										  boxstyle="round,pad=0.1",
										  facecolor='#70AD47', edgecolor='#4A8A2C', alpha=0.9)
			ax.add_patch(rect)

			text = format_hypothesis_short(g)
			ax.text(x, y_g, text, ha='center', va='center',
					fontsize=9, color='white', fontweight='bold')
	else:
		ax.text(6, y_g, "初始状态或空集", ha='center', va='center',
				fontsize=11, color='gray', style='italic')

	for i, s in enumerate(S):
		for j, g in enumerate(G):
			if is_general_than(g, s):
				x_s = start_x + i * spacing
				x_g = start_x + j * spacing
				x_s = max(1, min(11, x_s))
				x_g = max(1, min(11, x_g))

				# 绘制虚线连接
				ax.plot([x_s, x_g], [y_s + 0.1, y_g - 0.1],
						'gray', linestyle='--', linewidth=1, alpha=0.6)

	ax.annotate('', xy=(1.5, 6.5), xytext=(1.5, 3.5),
				arrowprops=dict(arrowstyle='<->', color='#FF6B6B', lw=2))
	ax.text(0.8, 5, '泛化\n方向', ha='center', va='center',
			fontsize=9, color='#FF6B6B', fontweight='bold')

	ax.text(11, 5, '更一般 →\n← 更特殊', ha='center', va='center',
			fontsize=9, color='gray', style='italic',
			bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.7))

	vs_size = len(S) * len(G) if S and G else 0
	ax.text(6, 0.5, f'变型空间大小 (估算): {vs_size} 个边界对',
			ha='center', fontsize=9, color='gray', style='italic',
			bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.7))

	plt.tight_layout()
	plt.show()


def generate_SG_with_history(examples):
	num_attrs = len(attr_values)
	S = [['$'] * num_attrs]
	G = [['?'] * num_attrs]

	history = []

	for idx, (e, label) in enumerate(examples):
		history.append(([h.copy() for h in S], [g.copy() for g in G], e, label))

		if label:
			G = [g for g in G if agree(g, e, True)]
			S = [min_generalize(s, e) for s in S]
			S = [s for s in S if any(is_special_than(s, g) for g in G)]
		else:
			S = [s for s in S if agree(s, e, False)]
			new_G = []
			for g in G:
				if agree(g, e, True):
					new_G.extend(min_specialize(g))
				else:
					new_G.append(g)
			G = [list(g) for g in set(tuple(g) for g in new_G)]
			G = [g for g in G if any(is_general_than(g, s) for s in S)]

	history.append(([h.copy() for h in S], [g.copy() for g in G], None, None))

	return S, G, history


def run_with_visualization(examples):
	print("=" * 60)
	print("开始候选消除算法（带可视化）")
	print("=" * 60)

	S, G, history = generate_SG_with_history(examples)

	try:
		draw_hypothesis_tree(S, G, "最终变型空间")
	except Exception as e:
		print(f"可视化出错: {e}")
		print("提示: 请确保 matplotlib 已正确安装")

	return S, G


if __name__ == "__main__":
	examples = [
		(['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], True),
		(['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True),
		(['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False),
		(['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True),
	]

	print("\n训练样例:")
	for i, (e, label) in enumerate(examples):
		print(f"  {i + 1}. {e} → {'正例' if label else '负例'}")

	S, G = generate_SG(examples)

	print("\n" + "=" * 60)
	print("最终结果:")
	print(f"S (最特殊边界): {S}")
	print(f"G (最一般边界): {G}")


	def print_version_space(S, G):
		print("\n变型空间 (Version Space):")
		for s in S:
			for g in G:
				if is_general_than(g, s):
					print(f"  {s} → {g}")


	print_version_space(S, G)

	print("\n" + "-" * 8)
	print("扩展任务")
	print("-" * 8)
	new_example = (['Sunny', 'Cold', 'Normal', 'Strong', 'Warm', 'Same'], False)
	examples.append(new_example)
	print(f"\n添加新负例: {new_example[0]}")
	S2, G2 = generate_SG(examples)
	print("\n加入负例后 S:", S2)
	print("加入负例后 G:", G2)

	print("\n" + "=" * 60)
	print("生成可视化图形...")
	print("=" * 60)
	run_with_visualization(examples)