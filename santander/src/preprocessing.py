from collections import namedtuple
import pandas

def get_data(additional_filter=True, load_submissions=False):
	df = pandas.read_csv('../data/train.csv', header=0)

	print('Initial Data size: {}'.format(df.shape))
	#df = df.drop(["ID"], axis=1)

	data, target = df.values[:,:-1], df.values[:,-1:] # We split the data

	series = namedtuple('Series', ['data', 'target'])
	series.data = data
	series.target = target.ravel()

	return series
