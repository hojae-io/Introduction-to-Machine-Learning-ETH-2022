# Ali
## Task 1
Task 1 was about predicting binary value.
The first part was pre-processing the data. For each age category, we take the median value for each column from all the patient inside this age span, and use it to fill the NaN of those same patients. The NaN cells left are filled using the mean obtained from all the patient for each column.

Then for each column for which we have a prediction to make, we create a matrix X with the training features. Those feature are the age, the temperature, the RRate etc... that were found through trials.
Also we took only the min and max of those values over the 12 hour period so we dont have too much data.
After that, we rescaled the columns between -1 and 1 (this was a friends advice to improve the score)
and finally everything is fed to the classifier.
sigmoid is used to get the final results.

## Task 2
## Task 3


# Damien
## Task 1

## Task 2

## Task 3
Using the prepared data constructed in task 2 we undergo a multioutput regression using the "MultiOutputRegressor" of the sklearn.multioutput library.



# Luca
## Task 1
The first task needed you to forecast a binary value. Pre-processing the data was the first step. We use the median value for each column from all the patients within this age range to fill the NaN with those same patients within each age group. For each column, the NaN cells are packed with the mean collected from all of the patients.
Then we build a matrix X with the training features for and column for which we need to make a prediction. These characteristics include age, temperature, RRate, and others that were discovered through trials.
We also only took the minimum and maximum of those values over a 12-hour duration to avoid having too many data.
After that, we rescaled the columns from -1 to 1 (a friend's suggestion to increase the score), and then fed it to the classifier.
The final findings are obtained using sigmoid.

## Task 2
In Task 2, we prepared the data by reducing the dimensionality and making it less sparse. By taking the median of all features in the age range [patient_age-3, patient_age+3] we were able to fill all the NaN datapoints with sensible data. By using the following statistics: (min, max, median, standard deviation) over each patient across all the 12hours subsequently we obtained more meaningful statistics. This enabled us to use XGBoost Classification with optimised hyper-parameters to predict the probability of sepsis in a patient (LABEL_sepsis).
Something we tried as well was to regress linearly over all the 12 hours for each feature and patient and get the slope as well as the intercept as an additional feature, though this overloaded our data with too many features.
## Task 3
With the same optimized, cleaned and engineered data, we did a multioutput regression by using sklearn's MultiOutputRegressor. The following features were predicted here: LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate.