import pandas
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple

def get_data(load_test=False):
	#Load and process the training data
	print('Loading and processing training data...')
	path = '../data/train.csv'
	values = process_data(path)

	#Split features and label
	data, target = values[:,:-1], values[:,-1:]
	series = namedtuple('Series', ['data', 'target'])
	series.data = data
	series.target = target.ravel()

	if (load_test):
		print('Loading and processing test data...')
		path = '../data/test.csv'
		test = process_data(path,proc_label=False)

	return [series,test]

def process_data(path,proc_label=True):
	df = pandas.read_csv(path, header=0)
	le = LabelEncoder()

	#Drop ID column
	df = df.drop(df.columns[0], axis=1)

	# Convert all the categorical data to numerical values
	df.workclass = le.fit_transform(df.workclass)
	df.education = le.fit_transform(df.education)
	df['marital-status'] = le.fit_transform(df['marital-status'])
	df.occupation = le.fit_transform(df.occupation)
	df.relationship = le.fit_transform(df.relationship)
	df.race = le.fit_transform(df.race)
	df.sex = le.fit_transform(df.sex)

	#Capital gain mapping (Decreases score)
	#df['capital-gain'] = df['capital-gain'].apply(map_capital_gain)

	#Native country mapping
	df['native-country'] = df['native-country'].apply(map_native_country)

	#Process the label if it's training data
	if(proc_label):
		df.income = le.fit_transform(df.income)

	return df.values

def map_native_country(country):
	#United-States or None
	if country in ['United-States','None']:
		return 0

	#Developed Countries
	if country in ['England','Canada','Germany','Japan','Poland','China','Italy', \
	'Mexico','Portugal','Ireland','France','Holand-Netherlands','Greece','Hungary', \
	'Scotland']:
		return 1
	#Underdeveloped Countries
	if country in ['Cambodia','Puerto-Rico','Outlying-US(Guam-USVI-etc)','India', \
	'South','Cuba','Iran','Honduras','Philippines','Jamaica','Vietnam','Dominican-Republic', \
	'Laos','Ecuador','Taiwan','Haiti','Columbia','Nicaragua','Thailand','Peru', \
	'Hong','Trinadad&Tobago','Guatemala','Yugoslavia','El-Salvador']:
		return 2


def map_capital_gain(amount):
	if(amount > 10000):
		return 4
	elif(amount > 5000):
		return 3
	elif(amount > 1000):
		return 2
	else:
		return 0
