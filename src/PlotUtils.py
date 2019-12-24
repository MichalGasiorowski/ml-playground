import matplotlib.pyplot as plt


def plot_history(history_dict, what_to_plot, title, xlabel='Epochs', ylabel='Accuracy'):
	# [(k1,l1),(k2,l2)]

	key1, plot_label_1 = what_to_plot[0]
	key2, plot_label_2 = what_to_plot[1]

	epochs = range(1, len(history_dict[key1]) + 1)
	plt.clf()  # clear figure

	plt.plot(epochs, history_dict[key1], 'bo', label=plot_label_1)
	plt.plot(epochs, history_dict[key2], 'b', label=plot_label_2)
	# plt.title('Training and validation accuracy')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	plt.show()