import numpy
from collections import namedtuple

def get_data(additional_filter=True, load_submissions=False):
	samples = numpy.genfromtxt('../data/train.csv', delimiter=',', skip_header=1)

	print('Initial Data size: {}'.format(samples.shape))

	data, target = samples[:,:-1], samples[:,-1:] # We split the data

	series = namedtuple('Series', ['data', 'target'])
	series.data = data
	series.target = target.ravel()

	return series
