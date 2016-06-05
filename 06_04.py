
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

XLIMIT = 10
YLIMIT = 10

FEATURE_LIST = ['x','y','accuracy','day','dow','hour']

def read_files(train_name, test_name):
	types = {'x': np.dtype(float), 'y' : np.dtype(float), 'accuracy': np.dtype(int), 'time': np.dtype(int), 'place_id': np.dtype(int)}
	
	train = pd.read_csv(train_name, dtype=types)
	test = pd.read_csv(test_name, dtype=types)

	return train, test


#def generate_cells:



def generate_features(train, test):

	#as verified, time is in minutes
	train['day'] = (train['time'] / (60.0 * 24.0)).astype(int)
	train['dow'] = train['day'] % 7
	train['hour'] = (train['time'] / 60.0).astype(int) % 24

	test['day'] = (test['time'] / (60.0 * 24.0)).astype(int)
	test['dow'] = test['day'] % 7
	test['hour'] = (test['time'] / 60.0).astype(int) % 24

	#split the map into several buckets
	cell_size = 2
	cell_total = (XLIMIT/cell_size) * (YLIMIT/cell_size)
	train['xbucket'] = (train['x'].astype(int) / cell_size).astype(int)
	train['ybucket'] = (train['y'].astype(int) / cell_size).astype(int)
	train['cellID'] = train['ybucket'] * (XLIMIT/ cell_size) + train['xbucket']

	test['xbucket'] = (test['x'].astype(int) / cell_size).astype(int)
	test['ybucket'] = (test['y'].astype(int) / cell_size).astype(int)
	test['cellID'] = test['ybucket'] * (XLIMIT/ cell_size) + test['xbucket']

	#normalize features	
	for f in FEATURE_LIST:
		f_mean = train[f].mean()
		f_std = train[f].std()
		train[f] = (train[f] - f_mean) / f_std
		test[f] = (test[f] - f_mean) / f_std


	return train, test, cell_total


def train_model_on_cell(train, test, numCells):

	for i in range(numCells):
		print "    cell %d ..." % i

		train_curr = train.loc[train.cellID == i].reset_index(drop = True)
		train_X = train_curr[FEATURE_LIST].as_matrix()
		train_Y = train_curr['place_id'].values

		test_index = test.loc[test.cellID == i].index
		test_label = test.loc[(test.cellID == i),'place_id'].values
		test_curr = test.loc[test.cellID == i].reset_index(drop = True)
		test_X = test_curr[FEATURE_LIST].as_matrix()

		rf = RandomForestClassifier(max_depth=10, n_estimators=10)
		rf.fit(train_X, train_Y)
		test_predict = rf.predict(test_X)

		result_df = pd.DataFrame({'index': test_index, 'label': test_label, 'predict_label': test_predict})
		print result_df


		print "    finished."
		break






if __name__ == '__main__':

	print "Reading files ..."
	train_df, test_df = read_files('small_train.csv', 'small_test.csv')
	print "    Finished."


	print "Generating features ..."
	train_df, test_df, cell_total = generate_features(train_df, test_df)
	print "    Finished."

	print "Training cell by cell ..."
	train_model_on_cell(train_df, test_df, cell_total)


