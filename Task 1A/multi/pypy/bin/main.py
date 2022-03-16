from multiprocessing import Pool

# import Sklearn packages
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold 

# import helper packages
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt
import numpy as np


# measuring time
import time
start_time = time.time()


# Our constants
lambdas = [0.1,1,10,100,200]
folds = 10
M = 10**6

# Read and split data
df = pd.read_csv("train.csv")

# Initialize root mean squared list (over lambdas)
# Iterate over lambdas


def meaning(i):
    # Initialize root mean squared list (over lambdas)
    main_rmses = [] # for each labdas
    # Iterate over lambdas
    for lam in lambdas:
        kf = KFold(n_splits=folds, shuffle=True)
        # Initialize root mean squared list (over folds)
        sub_rmses = []
        # Iterate over folds
        for train_index, test_index in kf.split(df):
            X_train, X_test = df.iloc[train_index,1:], df.iloc[test_index,1:]
            y_train, y_test = df.iloc[train_index,0], df.iloc[test_index,0]

            # Fit and predict Ridge regression
            clf = Ridge(alpha=lam, fit_intercept=False).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            sub_rmses.append(sqrt(mean_squared_error(y_test, y_pred)))
        main_rmses.append(mean(sub_rmses))
    return main_rmses

if __name__ == '__main__':
	a_pool = Pool()
	chunks = 100
	result = [0,0,0,0,0]
	N = M
	while (N>0):
		big_result = np.array(a_pool.map(meaning, range(chunks)))
		N-=chunks
		chunk_result = [sum(i) for i in zip(*big_result)]
		result = [result[i] + chunk_result[i] for i in range(len(result))]
	final_result = [x/M for x in result]
	print(final_result)
	print("--- %s seconds ---" % (time.time() - start_time))


	# Plot mean root meaned errors over lambda
	#plt.plot(lambdas,final_rmses)
	#plt.show()

	# Export results to output.csv
	#df_export = pd.DataFrame(final_rmses)
	#df_export.to_csv('output_intercept1_e3.csv', index=False, header=False)  

