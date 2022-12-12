import seaborn as sns
import matplotlib.pyplot as plt


def accuracyPlotter(log):
    """
    plot accuracy according to each classifier
    """
    sns.barplot(x='Accuracy', y='Classifier', data=log)

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()


def loglossPlotter(log):
    """
    plot log loss according to each classifier
    """
    sns.barplot(x='Log Loss', y='Classifier', data=log)

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.show()


def subPlotter121(log):
    """
    plot accuracy & log loss according to each classifier
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle('Classifier Showdown')

    sns.barplot(ax=axes[0], x='Accuracy', y='Classifier', data=log)
    axes[0].set_title('Accuracy')

    sns.barplot(ax=axes[1], x='Log Loss', y='Classifier', data=log)
    axes[1].set_title('LogLoss')

    plt.show()
