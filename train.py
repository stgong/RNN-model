from __future__ import print_function

import numpy as np
import helpers.command_parser as cp
import helpers.command_parser as parse
import helpers.early_stopping as EsParse
from helpers.data_handling import DataHandler



def training_command_parser(parser):
	parser.add_argument('--tshuffle', help='Shuffle sequences during training.', action='store_true')
	parser.add_argument('--extended_set', help='Use extended training set (contains first half of validation and test set).', action='store_true')
	parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
	parser.add_argument('--dir', help='Directory name to save model.', default='', type=str)
	parser.add_argument('--save', choices=['All', 'Best', 'None'], help='Policy for saving models.', default='Best')
	parser.add_argument('--metrics', help='Metrics for validation, comma separated', default='sps', type=str)
	parser.add_argument('--load_last_model', help='Load Last model before starting training.', action='store_true')
	parser.add_argument('--number_of_batches', help='number_of_batches', default='5000', type=str)
	parser.add_argument('--mpi', help='Max progress intervals', default=50000, type=int)
	parser.add_argument('--max_iter', help='Max number of iterations', default=10, type=int)
	parser.add_argument('--debug', help='Max training time in seconds', default= False, type=bool)
	parser.add_argument('--epochs', help='number of epochs', default=10, type=int) # 10 epoch for ml1m


def num(s):
	try:
		return int(s)
	except ValueError:
		return float(s)


def main():

	args = cp.command_parser(training_command_parser, cp.predictor_command_parser, EsParse.early_stopping_command_parser)
	predictor = parse.get_predictor(args)

	dataset = DataHandler(dirname=args.dataset, extended_training_set=args.extended_set, shuffle_training=args.tshuffle)

	predictor.prepare_networks(dataset.n_items)
	predictor.train(dataset,
		number_of_batches=num(args.number_of_batches),
		autosave=args.save,
		save_dir=dataset.dirname + "/models/" + args.dir,
		epochs=args.epochs,
		early_stopping=EsParse.get_early_stopper(args),
	    validation_metrics = args.metrics.split(','),
					debug = args.debug)


if __name__ == '__main__':
	main()
