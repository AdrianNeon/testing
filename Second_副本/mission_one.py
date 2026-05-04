def agree(h, e, elabel):
	covers = all(h[i] in (e[i], '?') for i in range(len(h)))

	if elabel:
		return covers
	else:
		return not covers

def find_S(examples, attributes):
	h = ['$'] * len(attributes)
	for idx, e in enumerate(examples):
		if not e[1]:
			continue

		covered = agree(h, e[0], True)

		if not covered:
			new_h = []
			for i in range(len(h)):
				if h[i] == '$':
					new_h.append(e[0][i])
				else:
					if h[i] != e[0][i]:
						new_h.append('?')
					else:
						new_h.append(h[i])
			h = new_h

	return h

def test_find_S_orders():
	attributes = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']

	examples_order1 = [
		(['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], True),
		(['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True),
		(['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False),
		(['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True),
	]

	examples_order2 = [
		(['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False),
		(['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True),
		(['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True),
		(['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], True),
	]

	result1 = find_S(examples_order1, attributes)
	result2 = find_S(examples_order2, attributes)

	print("Find-S Results:")
	print("Order 1 result:", result1)
	print("Order 2 result:", result2)
	print("Results are identical:", result1 == result2)
	print()

test_find_S_orders()