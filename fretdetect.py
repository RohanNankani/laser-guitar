import argparse

parser = argparse.ArgumentParser(description='Laser guitar fret fingering ML model generator and runner command line tool.')

command = parser.add_mutually_exclusive_group()
command.add_argument('--train', '-t', dest="train", help='Train the dataset', action='store_true')
command.add_argument('--predict', '-p', dest="predict", help='Use a trained model to predict the class of an image', type=str)

parser.add_argument('--model', '-m', required=True, help='File name for the TensorFlow model', type=str)
parser.add_argument('--quiet', '-q', help='Do not print info logging', action='store_true')
parser.add_argument('--epochs', '-e', help='Number of Epochs to train the data on', type=int, default=5)

args = parser.parse_args()
if args.train:
    import v3_fingers
    v3_fingers.train_model(args.model, args.epochs, verbose=not args.quiet)
elif args.predict:
    import v3_test
    v3_test.predict(args.predict, args.model)
else:
    print("error: please provide a command:")
    parser.print_help()