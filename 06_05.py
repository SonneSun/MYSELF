#Facebook Kaggle Competition
import time
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

XLIMIT = 10
YLIMIT = 10

#FEATURE_LIST = ['x','y','accuracy','day','dow','hour']
FEATURE_LIST = ['x','y','accuracy','dow','month','hour']

def read_files(train_name, test_name):
	types = {'row_id': np.dtype(int), \
	         'x': np.dtype(float),\
	         'y' : np.dtype(float), \
	         'accuracy': np.dtype(int), \
	         'time': np.dtype(int), \
	         'place_id': np.dtype(int)}
	
	train = pd.read_csv(train_name, dtype=types)
	test = pd.read_csv(test_name, dtype=types)

	return train, test



def generate_features(train, test):

	#as verified, time is in minutes
	train['day'] = (train['time'] / (60.0 * 24.0)).astype(int)
	train['dow'] = train['day'] % 7
	train['month'] = train['day'] % 30
	train['hour'] = (train['time'] / 60.0).astype(int) % 24

	
	# train.loc[((train['hour'] >= 0) & (train['hour'] < 8) ), 'tod'] = 1
	# train.loc[((train['hour'] >= 8) & (train['hour'] < 16) ), 'tod'] = 2
	# train.loc[((train['hour'] >= 16) & (train['hour'] < 24) ), 'tod'] = 3
	

	test['day'] = (test['time'] / (60.0 * 24.0)).astype(int)
	test['dow'] = test['day'] % 7
	test['month'] = test['day'] % 30
	test['hour'] = (test['time'] / 60.0).astype(int) % 24

	# test.loc[((test['hour'] >= 0) & (test['hour'] < 8) ), 'tod'] = 1
	# test.loc[((test['hour'] >= 8) & (test['hour'] < 16) ), 'tod'] = 2
	# test.loc[((test['hour'] >= 16) & (test['hour'] < 24) ), 'tod'] = 3

	#split the map into several buckets
	xcell_size = 0.5
	ycell_size = 1
	cell_total = int((XLIMIT/xcell_size) * (YLIMIT/ycell_size))

	train['x_adj'] = train['x']
	train.loc[(train['x_adj'] == XLIMIT),'x_adj'] = train['x_adj'] - 0.0001
	train['y_adj'] = train['y']
	train.loc[(train['y_adj'] == YLIMIT),'y_adj'] = train['y_adj'] - 0.0001

	train['xbucket'] = (train['x_adj'].astype(int) / xcell_size).astype(int)
	train['ybucket'] = (train['y_adj'].astype(int) / ycell_size).astype(int)
	train['cellID'] = train['ybucket'] * (XLIMIT/ xcell_size) + train['xbucket']

	test['x_adj'] = test['x']
	test.loc[(test['x_adj'] == XLIMIT),'x_adj'] = test['x_adj'] - 0.0001
	test['y_adj'] = test['y']
	test.loc[(test['y_adj'] == YLIMIT),'y_adj'] = test['y_adj'] - 0.0001

	test['xbucket'] = (test['x_adj'].astype(int) / xcell_size).astype(int)
	test['ybucket'] = (test['y_adj'].astype(int) / ycell_size).astype(int)
	test['cellID'] = test['ybucket'] * (XLIMIT/ xcell_size) + test['xbucket']

	#normalize features	
	# for f in FEATURE_LIST:
	# 	f_mean = train[f].mean()
	# 	f_std = train[f].std()
	# 	train[f] = (train[f] - f_mean) / f_std
	# 	test[f] = (test[f] - f_mean) / f_std


	return train, test, cell_total


def train_model_on_cell(train, test, numCells, th):

	for i in range(numCells):
		print "    cell %d ..." % i
		start = time.time()

		train_curr = train.loc[train.cellID == i].reset_index(drop = True)
		if len(train_curr) == 0:
			continue

		place_counts = train_curr.place_id.value_counts()
		mask = place_counts[train_curr.place_id.values] >= th
		train_curr = train_curr.loc[mask.values].reset_index(drop = True)


		train_X = train_curr[FEATURE_LIST].as_matrix()
		#weight = train_curr['accuracy'].values
		train_Y = train_curr['place_id'].values

		# le = LabelEncoder()
		# train_Y  = le.fit_transform(train_Y)

		test_index = test.loc[test.cellID == i].index
		test_label = test.loc[(test.cellID == i),'place_id'].values
		test_curr = test.loc[test.cellID == i].reset_index(drop = True)
		test_X = test_curr[FEATURE_LIST].as_matrix()

		# print len(train_curr)
		# print len(test_curr)

		#increase estimators won't influence too much
		rf = RandomForestClassifier(max_depth = 20, n_estimators = 30)  

		#add weight won't influence too much
		#rf.fit(train_X, train_Y, sample_weight = weight)
		rf.fit(train_X, train_Y)
		test_predict = rf.predict(test_X)
		
		# test_predict = np.argsort(test_predict, axis=1)[:,::-1][:,:3]
		# test_predict = le.inverse_transform(test_predict)
		

		result_df = pd.DataFrame({'index': test_index, 'place_id': test_label, 'predict': test_predict})
		print len(result_df.loc[result_df.place_id == result_df.predict]) / float(len(result_df))
		
		#test.loc[test_index,'predict'] = test_predict

		end = time.time()
		print "    finished. Uses %f s in time. " % (end - start) 

		break
		
	return test




if __name__ == '__main__':

	print "Reading files ..."
	train_df, test_df = read_files('small_train.csv', 'small_test.csv')
	print "    Finished."


	print "Generating features ..."
	train_df, test_df, cell_total = generate_features(train_df, test_df)
	print "    Finished."

	print "Training cell by cell. Total number is %d  ..." % cell_total
	test_df = train_model_on_cell(train_df, test_df, cell_total, 100)

	#print len(test_df.loc[test_df.place_id == test_df.predict]) / float(len(test_df))

	# result_df = test_df[['row_id','predict']]
	# result_df.columns = ['row_id', 'place_id']
	# result_df.to_csv('result.csv', index = False)


