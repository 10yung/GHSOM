import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import ExcelWriter
import openpyxl
import datetime
from clustering.som import SOM
from clustering.ghsom_test import GHSOM
import random




# #-----------------------------------Data pre-process-------------------------------------------------------
# # get  training data
# df = pd.read_excel('./data/WPG_data_test.xlsx')
#
# # -----------------------------------input data random pre-process------------------------------------
# df_ran = df.sample(frac=1)
#
# # define which title to be noimal
# df_nominal = df_ran.ix[:, ['Report Date', 'Customer', 'Type','Item Short Name', 'Brand', 'Sales']]
# # df_numerical_tmp = df_ran.ix[:, ['OH WK', 'OH FCST WK', 'BL WK', 'BL FCST WK', 'Last BL', 'Backlog', 'BL <= 9WKs', 'DC OH', 'On the way', 'Hub OH', 'Others OH', 'Avail.', 'Actual WK', 'FCST WK', 'Actual AWU', 'FCST AWU', 'FCST M', 'FCST M1', 'FCST M2', 'FCST M3']]
# # df_numerical = df_numerical_tmp.apply(pd.to_numeric, errors='coerce').fillna(-1)
#
#
#
# # get data dim to latter SOM prcess
# input_dim = len(df_numerical.columns)
# input_num = len(df_numerical.index)
#
#
# # -----------------------------------input data random pre-process------------------------------------
# # change data to np array (SOM accept nparray format)
# input_data = np.array(df_numerical)


# -----------------------------------tmp input data------------------------------------
input_data = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0]])


ghsom = GHSOM(2, 2, 3, len(input_data), input_data, 6, 0.3, 0.3)
ghsom.train(input_data)

#-----------------------------------SOM process-------------------------------------------------------

# #Train a 20x30 SOM with 400 iterations
# som = SOM(2, 2, 3, input_data, 6)
# print('training start : ' + str(datetime.datetime.now()))
# som.train(input_data)
#
#
# #Map datato their closest neurons
# mapped = som.map_vects(input_data)
# result = np.array(mapped)
#
# print(result)


#-------------------------------------Output format-----------------------------------------------------

# # output format
# output_np = np.concatenate((df_nominal, result), axis=1)
# output_pd = pd.DataFrame(data=output_np, columns=['Report Date', 'Customer', 'Type', 'Item Short Name', 'Brand', 'Sales', 'axis-x', 'axis-y'])
# # print(output_pd)


# # write to final csv
# output_pd.to_csv('./result/result.csv')
