===============================================================================
HIGH-LEVEL INFORMATION
-------------------------------------------------------------------------------
Project Goal : Create a pipeline that accomplishes 4 main things via a combination of Python and R scripts
[1] Retrieve financial time-series stock data corresponding to a specified ending date
[2] Perform feature engineering on this financial time-series data for one of two cases
    {Case 1} Obtain feature engineered data to use for predictive algorithm test/training purposes --> Contains X variables and a Y variable
        * In this case the outcome (Y-variable) is KNOWN
    {Case 2} Obtain feature engineered data to use for NEW prediction purposes
        * In this case the outcome (Y-variable) is UNKNOWN
        * In this case it is assumed that predictive algorithms have already been created
[3] Utilize feature engineered data (corresponding to Purpose 1 above) to TRAIN/TEST several predictive algorithms and QUANTIFY their performance
[4] Utilize feature engineered data (corresponding to Purpose 2 above) to make NEW predictions

General Prediction Model Overview : The input-output structure of all predictive algorithms used in this project is as follows
    - X-variables: These are cross-sectional features of a stock which are obtained by data engineering time-series data of price (open, close, low, high) and volume
        * X-variables can range from SMAs, correlations, to CAPM regression variables
    - Y-variable: The outcome variable is the N-day future return for a stock (could be the 3-day, 5-day, etc.)
     
Premise of Prediction Approach : The background/premise of the approach is as follows.
    - By using time-series data corresponding to a specified end date for N stocks with known outcome (Y-variable), I can data engineer M cross-sectional features for each stock, which results in an N x M data array
    - This N x M data array can be used to test/train predictive algorithms corresponding to the specified end date of the time-series data which was used to create the N x M array
        * We can also quantify the performance of the predictive algorithms on testing/training data via statistical metrics like accuracy, F1 score, etc.
    - These predictive algorithms can be used to predict future return for stocks with a specified end date that is LATER than the one used to create the predictive models
        * Suppose the predictive model was created based on time-series data of several stocks with specified end-date of 10/12/2021
        * Then I am asserting that I could use the predictive model to predict NEW returns for stock with a specified end-date of something like 10/19/2021
        * Obviously the predictive potential of a financial model is more viable in the time frame for which it is created but the whole point is making future predictions
        * The predictive model does not capture causal relationships, it utilizes correlations to construct predictive algorithms which is an inherent limitation since those causal relationships can change from time period to time period

Info about End Date : This is important info about "End Date" which is repeatedly mentioned above
    - The end date serves as a reference point for cross-sectional feature creation
    - For example if the end date is 10/12/2021, then the various cross sectional features of a stock such as SMAs, correlations are with respect to 10/12/2021
    - If the end date is 10/12/2021 then the N-day return corresponds to N trading days after 10/12/2021
    - NOTE that there is a difference between End Date in Case 1 vs Case 2, refer to the Python file for details about that, what is provided here is a broad overview

===============================================================================
FUNCTION INFORMATION 
-------------------------------------------------------------------------------
StockFeatureEngineering.py : This Python file is used to create the M x N data array mentioned above for two possible cases
{Case 1} Obtain feature engineered data for several stocks to use for predictive algorithm test/training purposes
    - In this case the outcome (Y-variable) is KNOWN, so the M x N data array will also have an additional column with this output variable
    - Inputs:
        * CSV file with list of several tickers to sample from 
        * N, specifying max number of tickers to sample for the M x N data array creation 
        * Specified end-date (reference point for constructing cross-sectional features)
    - Outputs:
        * CSV file with M x N data array (and a column for Y-variable)
{Case 2} Obtain feature engineered data for several stocks to use for NEW future return prediction purposes
    - In this case the outcome (Y-variable) is UNKOWN, so the M x N data array will purely contain cross-sectional features for each stock
    - Inputs: 
        * CSV file with list of tickers you are interested in predicting future return of
        * Specified end-date (reference point for constructing cross-sectional features)
    - Outputs: 
        * CSV file with M x N data array (no column for Y-variable)

-------------------------------------------------------------------------------
StockPredictionML.R : This R script is used in a couple possible ways corresponding to the cases above
{Case 1} In this case, this script is used to construct several predictive algorithms based on the testing/training data obtained from the Python script
    - Primary input is the CSV file with the M x N data array obtained by using the Python file for Case 1 (Y-variable known)
    - In this case, the performance of each predictive algorithm on testing and training data is quantified and output in a CSV file
{Case 2} In this case, this script is used to predict future returns for specified stocks
    - Primary input is the CSV file with the M x N data array obtained by using the Python file for Case 2 (Y-variable unknown)
    - In this case the predictions each algorithm makes for every stock of interest is output in a CSV file