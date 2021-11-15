import matplotlib.pyplot as plt
import pickle


def visualize(filename, xvalues, yvalues, title='figure', xlabel='x_label', ylabel='y_label'):
    """
        Helper functions for analysis
        Draws plot then save as image
    """
    plt.figure()
    plt.scatter(xvalues, yvalues)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    ### TODO: WRITE YOUR CODE HERE. ############################################
    xs, ys = [], []

    # For the problem 3.1
    with open('./problem3_1.pickle', 'rb') as f:
        xs, ys = pickle.load(f)
    f.close()

    visualize('problem3_1.png', xs, ys, r"Effect of the infection rate $\beta$", r"$\beta$ values", "# of recovered nodes") # Example

    # For the problem 3.2
    with open('./problem3_2_1.pickle', 'rb') as f:
        xs, ys = pickle.load(f)
    f.close()

    visualize('problem3_2_1.png', xs, ys, r"$\beta=0.005, \delta=0.8$", r"# of $N_0$", "# of recovered nodes") # Example

    with open('./problem3_2_2.pickle', 'rb') as f:
        xs, ys = pickle.load(f)
    f.close()

    visualize('problem3_2_2.png', xs, ys, r"$\beta=0.02, \delta=0.6$", r"# of $N_0$", "# of recovered nodes") # Example

    ############################################################################