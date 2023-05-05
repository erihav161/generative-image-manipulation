from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import argparse

def main(args):

    # Get filenames from args
    train_file = args.train_filename
    test_file = args.test_filename

    # Load the training and test loss dictionaries
    train_loss = load(open(train_file + '.pkl', 'rb'))
    if test_file:
        test_loss = load(open(test_file + '.pkl', 'rb'))
    
    num_epochs = list(train_loss)[-1]

    # Retrieve each dictionary's values
    train_values = train_loss.values()
    if not test_file:
        test_values = range(num_epochs)
    else:
        test_values = test_loss.values()
    
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, num_epochs + 1)

    # Plot and label the training and test loss values
    plt.plot(epochs, train_values, label='Discriminator Loss')
    plt.plot(epochs, test_values, label='Generator Loss')
    
    # Add a title and axis labels
    plt.title('Generator and Discriminator Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Set the tick locations
    plt.xticks(arange(0, num_epochs + 1, 100))
    
    # Display the plot
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Loss plotter')
    parser.add_argument('--train-filename', type=str, default='train_loss',
                        help='name of the pickle file containing train loss')
    parser.add_argument('--test-filename', type=str, default='test_loss',
                        help='name of the pickle file containing train loss')
    
    args = parser.parse_args()
    main(args)