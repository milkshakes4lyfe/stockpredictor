# -*- coding: utf-8 -*-
"""
StockFeatureEngineering.py  
@author: Chaitanya Panuganti
"""

# Importing Libraries ########################################################
import yfinance as yf
import pandas as pd
import statistics as st
from scipy.stats import pearsonr
from datetime import date, timedelta
import random
import time 
from threading import Thread
from sklearn.linear_model import LinearRegression
import numpy as np

# Global Variables ###########################################################
# Global variables to help thread execution to time-limit yfinance API calls
globalNewExtract = 0    
globalFcnFinished = False

# Global variable to dictate which of the 2 use-cases the script is run for
boolIsTraining = True

# Helper Functions ###########################################################
'''
featureExtract: This function creates a dictionary containing constructed data
features for a particular stock based on the inputs provided. 

Inputs:
    - ticker: A string with the stock's ticker symbol
    - dateEnd: A datetime object (e.g. date(2020,04,20)) for the end date
    - boolIsTraining: A boolean telling whether function is being called for 
      use on training/validation data or not. 
    - spyData: Downloaded time-series data for SPY from Yahoo finance. 
    - shvData: Downloaded time-series data for SHV from Yahoo finance. 
Outputs:
    - Dictionary with various constructed data features for that specific 
      stock. 
      
If boolIsTraining = False, the function uses the provided end date and creates
all data features with that end date in mind. 

If boolIsTraining = True, then the function provides the 5 day return assuming 
the end date, but it makes the constructed data features assuming a date 5  
days before the end date that is provided. 

The intention is that 5 day return is the dependent variable and the other 
features in the dictionary are predictor variables. That is why 5 day return 
is only created for training/validation data, when boolIsTraining = True.

'''
def featureExtract(ticker,dateEnd,boolIsTraining,spyData,shvData,nDayReturn):
    global globalNewExtract 
    global globalFcnFinished 
    
    globalFcnFinished = False
     
    ## Downloading time-series data from Yahoo Finance
    # Time-series data for the target ticker symbol
    stockData = tickerTsDownload(ticker,dateEnd)
    
    ## Assigning columns from time-series data (numpy format) to variables 
    # Time-series variables for the target ticker symbol
    price = np.array(stockData.Close)
    volume = np.array(stockData.Volume)
    openPrice = np.array(stockData.Open)
    intradayReturn = price/openPrice - 1
    totalDailyReturn = price[1::]/price[0::-1] - 1
    # Time-series variables for the reference 
    spyPrice = np.array(spyData.Close)
    shvPrice = np.array(shvData.Close)
    spyTotalDailyReturn = spyPrice[1::]/spyPrice[0::-1] - 1
    shvTotalDailyReturn = shvPrice[1::]/shvPrice[0::-1] - 1
    
    ## Index tracking variables to be used in feature engineering
    # If data is for training pick the index that separates training and test
    nOffset = nDayReturn*(boolIsTraining==1)
    idx = len(stockData)-1-nOffset # Index of nOffset days before last date
    end = len(stockData)-1 # Index of last date
    
    ## Feature engineering on ts data to get cross-sectional features for ticker
    # Creating dictionary to store data features in 
    stockDict = dict()
    # Intraday price return averages for ticker
    stockDict['IntradayReturnSMA5'] = st.mean(intradayReturn[idx-4:idx+1]) 
    stockDict['IntradayReturnSMA10'] = st.mean(intradayReturn[idx-9:idx+1]) 
    stockDict['IntradayReturnSMA15'] = st.mean(intradayReturn[idx-14:idx+1]) 
    stockDict['IntradayReturnSMA20'] = st.mean(intradayReturn[idx-19:idx+1]) 
    # Total daily price return averages for ticker
    stockDict['TotalDailyReturnSMA5'] = st.mean(totalDailyReturn[idx-4:idx+1]) 
    stockDict['TotalDailyReturnSMA10'] = st.mean(totalDailyReturn[idx-9:idx+1]) 
    stockDict['TotalDailyReturnSMA15'] = st.mean(totalDailyReturn[idx-14:idx+1]) 
    stockDict['TotalDailyReturnSMA20'] = st.mean(totalDailyReturn[idx-19:idx+1]) 
    # Price moving day averages for ticker 
    stockDict['PriceSMA5'] = st.mean(price[idx-4:idx+1]) 
    stockDict['PriceSMA10'] = st.mean(price[idx-9:idx+1]) 
    stockDict['PriceSMA15'] = st.mean(price[idx-14:idx+1]) 
    stockDict['PriceSMA20'] = st.mean(price[idx-19:idx+1]) 
    # Price SMA Comparisons 
    stockDict['PriceSMA5gtSMA10'] = int(stockDict['PriceSMA5'] > stockDict['PriceSMA10'])
    stockDict['PriceSMA10gtSMA15'] = int(stockDict['PriceSMA10'] > stockDict['PriceSMA15'])
    stockDict['PriceSMA15gtSMA20'] = int(stockDict['PriceSMA15'] > stockDict['PriceSMA20'])
    stockDict['PriceSMA5gtSMA15'] = int(stockDict['PriceSMA5'] > stockDict['PriceSMA15'])
    stockDict['PriceSMA5gtSMA20'] = int(stockDict['PriceSMA5'] > stockDict['PriceSMA20'])
    stockDict['PriceSMA10gtSMA20'] = int(stockDict['PriceSMA10'] > stockDict['PriceSMA20'])
    # Normalized price moving day averages for ticker
    stockDict['NormPriceSMA5'] = st.mean(price[idx-4:idx+1]) / price[idx]
    stockDict['NormPriceSMA10'] = st.mean(price[idx-9:idx+1]) / price[idx]
    stockDict['NormPriceSMA15'] = st.mean(price[idx-14:idx+1]) / price[idx]
    stockDict['NormPriceSMA20'] = st.mean(price[idx-19:idx+1]) / price[idx]
    # Volume moving day averages for ticker
    stockDict['VolSMA5'] = st.mean(volume[idx-4:idx+1])
    stockDict['VolSMA10'] = st.mean(volume[idx-9:idx+1]) 
    stockDict['VolSMA15'] = st.mean(volume[idx-14:idx+1]) 
    stockDict['VolSMA20'] = st.mean(volume[idx-19:idx+1]) 
    # Volume SMA Comparisons 
    stockDict['VolSMA5gtSMA10'] = int(stockDict['VolSMA5'] > stockDict['VolSMA10'])
    stockDict['VolSMA10gtSMA15'] = int(stockDict['VolSMA10'] > stockDict['VolSMA15'])
    stockDict['VolSMA15gtSMA20'] = int(stockDict['VolSMA15'] > stockDict['VolSMA20'])
    stockDict['VolSMA5gtSMA15'] = int(stockDict['VolSMA5'] > stockDict['VolSMA15'])
    stockDict['VolSMA5gtSMA20'] = int(stockDict['VolSMA5'] > stockDict['VolSMA20'])
    stockDict['VolSMA10gtSMA20'] = int(stockDict['VolSMA10'] > stockDict['VolSMA20'])
    # Normalized volume moving day averages for ticker
    stockDict['NormVolSMA5'] = st.mean(volume[idx-4:idx+1]) / volume[idx]
    stockDict['NormVolSMA10'] = st.mean(volume[idx-9:idx+1]) / volume[idx]
    stockDict['NormVolSMA15'] = st.mean(volume[idx-14:idx+1]) / volume[idx]
    stockDict['NormVolSMA20'] = st.mean(volume[idx-19:idx+1]) / volume[idx]
    # Price-volume correlations over day ranges for ticker
    stockDict['PriceVolCorr5'] = pearsonr(price[idx-4:idx+1],volume[idx-4:idx+1])[0]
    stockDict['PriceVolCorr10'] = pearsonr(price[idx-9:idx+1],volume[idx-9:idx+1])[0]
    stockDict['PriceVolCorr15'] = pearsonr(price[idx-14:idx+1],volume[idx-14:idx+1])[0]
    stockDict['PriceVolCorr20'] = pearsonr(price[idx-19:idx+1],volume[idx-19:idx+1])[0]
    # Standard deviation of price over day ranges for ticker
    stockDict['PriceSTDV5'] = st.stdev(price[idx-4:idx+1]) 
    stockDict['PriceSTDV10'] = st.stdev(price[idx-9:idx+1])
    stockDict['PriceSTDV15'] = st.stdev(price[idx-14:idx+1]) 
    stockDict['PriceSTDV20'] = st.stdev(price[idx-19:idx+1])
    # Normalized standard deviation of price over day ranges for ticker
    stockDict['NormPriceSTDV5'] = st.stdev(price[idx-4:idx+1]) / st.mean(price[idx-4:idx+1])
    stockDict['NormPriceSTDV10'] = st.stdev(price[idx-9:idx+1]) / st.mean(price[idx-9:idx+1])
    stockDict['NormPriceSTDV15'] = st.stdev(price[idx-14:idx+1]) / st.mean(price[idx-14:idx+1])
    stockDict['NormPriceSTDV20'] = st.stdev(price[idx-19:idx+1]) / st.mean(price[idx-19:idx+1])
    # Standard deviation of daily return over day ranges for ticker
    stockDict['IntradayReturnSTDV5'] = st.stdev(intradayReturn[idx-4:idx+1]) 
    stockDict['IntradayReturnSTDV10'] = st.stdev(intradayReturn[idx-9:idx+1])
    stockDict['IntradayReturnSTDV15'] = st.stdev(intradayReturn[idx-14:idx+1]) 
    stockDict['IntradayReturnSTDV20'] = st.stdev(intradayReturn[idx-19:idx+1]) 
    # CAPM regression to get alpha and beta for the ticker
    Y = totalDailyReturn - shvTotalDailyReturn  # Excess return of ticker
    Y = Y.reshape(-1,1)
    X = spyTotalDailyReturn - shvTotalDailyReturn  # Excess return of S&P500 
    X = X.reshape(-1,1)
    linRegModel = LinearRegression().fit(X,Y)
    stockDict['Beta'] = float(linRegModel.coef_)
    stockDict['Alpha'] = float(linRegModel.intercept_)
    # If training data compute future 5 day return for training/validation 
    if (boolIsTraining):
        stockDict['Return'] = price[end]/price[idx] - 1
    ## Return output data
    globalNewExtract = stockDict
    globalFcnFinished = True
    return

'''
loadTickers: This function takes in a filename (in same folder as this file) 
that contains many ticker symbols and sample number N. It returns a list of 
N random tickers from that filename. 
'''
def loadTickers(tickersCSV,N):
    # Loading CSV of potential ticker symbols we want to extract features from
    allTickers = pd.read_csv(tickersCSV,header=None)
    allTickers = allTickers[0].tolist()
    # Shuffling the ticker symbols 
    for i in range(50):
        random.shuffle(allTickers)  
        
    # Sample list of N tickers from the potential ticker symbols
    N = min(len(allTickers),N)
    tickers = random.sample(allTickers,N)
    return tickers

'''
tickerTsDownload: This function returns time-series data for a stock ticker 
symbol by downloading it from the Yahoo finance server.   
'''
def tickerTsDownload(ticker,dateEnd):
    ## Computing a start date 35 days before dateEnd (for enough margin)
    dateStart = dateEnd - timedelta(days=35)
    
    ## Downloading and returning time-series data for the reference ticker
    referenceData = yf.download(ticker,dateStart.isoformat(),dateEnd.isoformat(),interval='1d')
    # If stockData last row date equals dateEnd remove it to avoid nan values
    lastDate = list(referenceData.index)[len(referenceData)-1].to_pydatetime().date()
    if lastDate == dateEnd:
        referenceData = referenceData.iloc[:-1,:]
    return referenceData

# Main Functions #############################################################
'''
main: This is the main function that handles the creation of a feature 
engineered dataset for multiple stocks over a time interval. The purpose is to 
generate a dataset that can be used for training/testing of prediction models. 

Inputs:
    - tickersCSV: Source file with list of possible ticker symbols
    - destinationCSV: Destination file to write data engineered dataset to
    - dateEnd: The end date to compute feature engineered data around 
    - N: Number of tickers to sample form the source file 
    
Output: No explicit output, writes feature engineered table to destinationCSV. 
'''
def main(tickersCSV,destinationCSV,dateEnd,N,boolIsTraining,nDayReturn):
    # Sample list of N tickers from the potential ticker symbols
    tickers = loadTickers(tickersCSV,N)   
    # Initializing lists to store feature vectors and tickers actually used
    data = []
    tickersUsed = []
    
    # Downloading the SPY and SHV reference time-series data
    spyData = tickerTsDownload('SPY',dateEnd)
    shvData = tickerTsDownload('SHV',dateEnd)
    
    # Iterating through tickers and extracting features
    for i in range(len(tickers)):
        # Adding a pause after every 10 ticker symbols to prevent server blockage
        if i%10 == 0:
            time.sleep(3)
        # Executing feature extraction function via a thread w/timeout limit
        p = Thread(target = featureExtract, args = [tickers[i],dateEnd,boolIsTraining,spyData,shvData,nDayReturn])
        p.start()
        p.join(timeout = 3)   
        # Add engineered data for ticker if featureExtract successfully runs
        if globalFcnFinished:   
            newExtraction = globalNewExtract
            data.append( list(newExtraction.values()) )
            tickersUsed.append(tickers[i])
    
    # Creating the Pandas data frame for extracted data and saving to CSV
    keys = list(newExtraction.keys()) # Obtaining the name of the columns
    data = pd.DataFrame(data,index=tickersUsed,columns=keys)
    data.to_csv(destinationCSV)
    return 

# Function Execution #########################################################
'''
Extract cross-sectional data for many stocks during same time interval.
    - Requires user to specify a source file with list of possible ticker 
      symbols, a destination file to write the cross-sectional feature 
      engineered data, an end date, and the number of tickers (N) to sample
      from the source file. 
      
'''

if not boolIsTraining: 
    # Extraction cross-sectional data for multiple stocks in same time period for
    # prediction purposes
    tickersCSV = 'Tickers4Prediction.csv'
    destinationCSV = 'CrossSectionalData4Prediction.csv'
    dateEnd = date(2021,7,5)
    N = 5000
    nDayReturn = 3
    main(tickersCSV,destinationCSV,dateEnd,N,boolIsTraining,nDayReturn)
else: 
    # Extracting cross-sectional data for multiple stocks in same time period for
    # training data creation purposes 
    tickersCSV = 'Tickers4TrainingData.csv'
    destinationCSV = 'StockCrossSectionalDataForTraining.csv'
    dateEnd = date(2021,7,5)
    N = 5
    nDayReturn = 3
    main(tickersCSV,destinationCSV,dateEnd,N,boolIsTraining,nDayReturn)