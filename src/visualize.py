import matplotlib.pyplot as plt
import re
import sys

def show_graph(path):
	f = open(path, 'r')
	val = [0, 0]
	loss = []
	epochs = []
	for i, line in enumerate(f):
		if i == 0:
			continue
		else:
			values = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
			if i > 3 and (i - 3) % 2 != 0:
				val.append(values[0])
			else:
				epochs.append(values[0])
				loss.append(values[1])
	#plt.plot(epochs, loss)
	plt.plot(epochs, val)
	plt.show()
	
if len(sys.argv) > 1:
	show_graph(sys.argv[1])