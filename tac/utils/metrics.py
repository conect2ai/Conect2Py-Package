import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


# ------------------------------
# STATISTICAL PROPERTIES
# ------------------------------
def calc_statistics(eval_df):
    """
    Calculate the basic statistics of the evaluation dataframe. 
    The dataframe must contain the columns 'original' and 'decompressed'.
    
    
    Returns a dataframe with the statistics of the evaluation dataframe. Including the columns:
    (min, max, mean, median, std, skewness, kurtosis)
    """
    try:
        # check if eval_df is a dataframe
        if not isinstance(eval_df, pd.DataFrame):
            raise ValueError('Eval_df must be a dataframe')
        
        # check if eval_df is a dataframe with the correct columns
        if not all(col in eval_df.columns for col in ['original', 'decompressed']):
            raise ValueError('Eval_df must contain the columns "original" and "decompressed"')

        df = eval_df[['original', 'decompressed']]

        if df.empty:
            raise ValueError('Input DataFrame is empty, cannot calculate statistics')

        # aggregate the basic stats
        basic_stats = df.agg(
            ['min', 'max', 'mean', 'median', 'std']
        )
        # calculate skewness and kurtosis
        skew = df.skew(skipna=False)
        kurt = df.kurtosis(skipna=False)

        # combine and return the dataframes
        final_stats = basic_stats.T.join([skew, kurt])
        final_stats.columns = ['min', 'max', 'mean',
                            'median', 'std', 'skewness', 'kurtosis']

        return final_stats
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    
    except AttributeError as ae:
        print(f"AttributeError: {ae}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# ------------------------------
# COMPRESSION RATE | SAMPLE REDUCTION
# ------------------------------
def file_size_compression_rate(original, compressed, save_to_csv=False):
    """
    Calculate the compression rate between the original and compressed series using the CSV file size
    
    This function saves the original and compressed series as CSV files to evaluate the file size compression using the operational system path.

    Returns a tuple with the compression rate, compression factor, original size and compressed size.
    """
    try:
        # check if original and compressed are dataframes
        if not isinstance(original, pd.Series) or not isinstance(compressed, pd.Series):
            return ValueError('Original and Compressed params expected to be dataframe objects')
    

        original_file_path = './_tmp_original.csv'
        compressed_file_path = './_tmp_compressed.csv'

        # save the series as csv to evaluate the file size compression
        original.to_csv(original_file_path)
        compressed.dropna().to_csv(compressed_file_path)

        # check if the files were created
        if not os.path.exists(original_file_path) or not os.path.exists(compressed_file_path):
            raise ValueError('Error creating the temporary files')

        original_file_stats = os.stat(original_file_path)
        compressed_file_stats = os.stat(compressed_file_path)

        original_size = original_file_stats.st_size
        compressed_size = compressed_file_stats.st_size

        comp_factor = original_size / compressed_size
        comp_rate = 1 - (compressed_size / original_size)

        if not save_to_csv:
            os.remove(original_file_path)
            os.remove(compressed_file_path)

        return comp_rate, comp_factor, original_size, compressed_size
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def sample_reduction_rate(original, compressed):
    """Calculate the number of sample reduction between the original and compressed series
    
    This function uses the length of the series to evaluate the sample reduction rate.

    Returns a tuple with the reduction rate, reduction factor, original length and compressed length.
    """
    try:
        # check if original and compressed are not empty
        if original.empty or compressed.empty:
            raise ValueError('Original and Compressed params expected to be not empty')

        original_length = len(original)
        compressed_length = len(compressed)

        red_factor = original_length / compressed_length
        red_rate = 1 - (compressed_length / original_length)

        return red_rate, red_factor, original_length, compressed_length
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# ------------------------------
# ERRORS
# ------------------------------

def MSE_score(original, decompressed):
    """Calculates and returns the Mean Squared Error (MSE) score between the original and decompressed series"""
    try:
        #check if original and decompressed are not empty
        if original.empty or decompressed.empty:
            raise ValueError('Original and Decompressed params expected to be not empty')
        
        mse = mean_squared_error(original, decompressed)
        return mse
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def RMSE_score(original, decompressed):
    """Calculates and returns the Root Mean Squared Error (RMSE) score between the original and decompressed series"""
    try:
        #check if original and decompressed are not empty
        if original.empty or decompressed.empty:
            raise ValueError('Original and Decompressed params expected to be not empty')
        
        rmse = mean_squared_error(original, decompressed, squared=False)
        return rmse
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def NRMSE_score(original, decompressed):
    """Calculates and returns the Normalized RMSE score between the original and decompressed series"""
    try:
        #check if original and decompressed are not empty
        if original.empty or decompressed.empty:
            raise ValueError('Original and Decompressed params expected to be not empty')

        nrmse = RMSE_score(original, decompressed)/np.std(original)
        return nrmse
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def MAE_score(original, decompressed):
    """Calculates and returns the Mean Absolute Error (MAE) score between the original and decompressed series"""
    try:
        #check if original and decompressed are not empty
        if original.empty or decompressed.empty:
            raise ValueError('Original and Decompressed params expected to be not empty')

        mae = mean_absolute_error(original, decompressed)
        return mae
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def PSNR_score(original, decompressed):
    """Calculates and returns the Peak Signal-to-Noise Ratio (PSNR) score between the original and decompressed series"""
    try:
        # check if original and decompressed are not empty
        if original.empty or decompressed.empty:
            raise ValueError('Original and Decompressed params expected to be not empty')
        
        max_val = np.max(original)
        min_val = np.min(original)
        mse = MSE_score(original, decompressed)
        psnr = 10*np.log10((max_val - min_val)**2/mse)
        return psnr
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def NCC_score(original, decompressed):
    """Calculates and returns the Normalized Cross Correlation (NCC) score between the original and decompressed series"""
    try:
        #check if original and decompressed are not empty
        if original.empty or decompressed.empty:
            raise ValueError('Original and Decompressed params expected to be not empty')

        norm_original = (original - np.mean(original)) / np.std(original)
        norm_decompressed = (decompressed - np.mean(decompressed)) / np.std(decompressed)

        length = len(original)

        ncc =  (1.0/(length-1)) * np.sum(norm_original*norm_decompressed)
        return ncc
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# ------------------------------
# PARAMETER OPTIMIZATION
# ------------------------------
def compression_f_score(rate, similarity, beta=1):
    """Calculate the harmonic mean between the compression error and compression rate"""
    # note: if error is used intead of similarity, use (1 - error)
    # x2 is the metric tha has higher weight (its beta times more important than x1)
    x1 = rate
    x2 = similarity

    # comp_criteria = (2 * rate *  (1 - error)) / (rate + (1 - error))
    cf_beta_score = (1 + beta**2) * ((x1 * x2) / (((beta**2)*x1) + x2))

    return cf_beta_score


# ------------------------------
# COMPRESSION REPORT
# ------------------------------
def get_compression_report(original, compressed, decompressed, rounding=4, cf_score_beta=1):
    """
    Get the compression report of the run.

    Input:
        - original: original series
        - compressed: compressed series
        - decompressed: decompressed series
        optional:
            - rounding: number of decimal places to round the results [default: 4]
            - cf_score_beta: beta parameter of the CF-Score [default: 1]

    Output:
    - Dictionary with the compression report including the following metrics: 
        - original_length, compressed_length, original_file_size, compressed_file_size,
        sample_reduction_rate, sample_reduction_factor, file_compression_rate, file_compression_factor and all
        error metrics (mse, rmse, nrmse, mae, psnr, ncc, cf_score)

    """
    try:
        # check if original, compressed and decompressed are not empty
        if original.empty or compressed.empty or decompressed.empty:
            raise ValueError('Original, Compressed or Decompressed are empty. Params expected to be not empty in order to calculate the compression report.')

        # evaluate compression and reduction rates
        reduction_rate, reduction_factor, original_length, compressed_length = sample_reduction_rate(original, compressed.dropna())
       
        compression_rate, compression_factor, original_file_size, compressed_file_size = file_size_compression_rate(original, compressed.dropna())
        
        # evaluate errors
        #comp_error = compression_error_score(original, decompressed)
        mse = MSE_score(original, decompressed)
        rmse = RMSE_score(original, decompressed)
        nrmse = NRMSE_score(original, decompressed)
        mae = MAE_score(original, decompressed)
        psnr = PSNR_score(original, decompressed)
        ncc = NCC_score(original, decompressed)
        cf_score = compression_f_score(
            compression_rate,
            ncc,
            beta=cf_score_beta
        )

        report = {
            'original_length': original_length,
            'compressed_length': compressed_length,
            'original_file_size': original_file_size,
            'compressed_file_size': compressed_file_size,
            'sample_reduction_rate': round(reduction_rate, rounding),
            'sample_reduction_factor': round(reduction_factor, rounding),
            'file_compression_rate': round(compression_rate, rounding),
            'file_compression_factor': round(compression_factor, rounding),
            'mse': round(mse, rounding),
            'rmse': round(rmse, rounding),
            'nrmse': round(nrmse, rounding),
            'mae': round(mae, rounding),
            'psnr': round(psnr, rounding),
            'ncc': round(ncc, rounding),
            'cf_score': round(cf_score, rounding),
        }
        return report
    
    except ValueError as ve:
        print(f"ValueError occurred while running 'get_compression_report': ", str(ve))


def print_compression_report(report, model_name='Model', model_params=[], cf_score_beta=1):
    """
    Based on the compression report, prints the compression report of the run.

    Input:
        - report: compression report (use get_compression_report() to generate the report)

        optional:
            - model_name: name of the model used in the run [default: 'Model']
            - model_params: parameters used in the models [default: {}]
            - cf_score_beta: beta parameter of the CF-Score [default: 1]
    """
    try: 
        # check if report is a dictionary
        if not isinstance(report, dict):
            raise ValueError('Please provide a dictionary as report, try using get_compression_report() before printing the report.')
        
        # check if report is not empty
        if not report:
            raise ValueError('Report is empty, please provide a valid report, try to use get_compression_report() before printing the report.')

        print('\n# RUN INFO #')
        print('- Model: ', model_name)
        if model_params:
            # check if model_params is a list
            if not isinstance(model_params, list):
                raise ValueError('Please provide a list as model_params')
            
            # get the model params from the list 
            print('- Optimal Params: ', model_params)

        print('- CF-Score Beta: ', cf_score_beta)

        print('\n# RESULTS #')
        reduction_factor = report['sample_reduction_factor']
        reduction_rate = report['sample_reduction_rate']
        compression_factor = report['file_compression_factor']
        compression_rate = report['file_compression_rate']

        # evaluate compression and reduction rates
        print('\nSAMPLES NUMBER reduction')
        print('- Original length: ', report['original_length'], ' samples')
        print('- Reduced length: ', report['compressed_length'], ' samples')
        print(f'- Samples reduced by a factor of {round(reduction_factor, 2)} times')
        print(f'- Sample reduction rate: {round(reduction_rate*100, 2)}%')
        print('\nFILE SIZE compression')
        print('- Original size: ', report['original_file_size'], ' Bytes')
        print('- Compressed size: ', report['compressed_file_size'], ' Bytes')
        print(f'- file compressed by a factor of {round(compression_factor, 2)} times')
        print(f'- file compression rate: {round(compression_rate*100, 2)}%')

        # errors
        print('\nMETRICS')
        #print(f'- Compression error: {round(comp_error, 4)}')
        print('- MSE: ', report['mse'])
        print('- RMSE: ', report['rmse'])
        print('- NRMSE: ', report['nrmse']),
        print('- MAE: ', report['mae']),
        print('- PSNR: ', report['psnr'])
        print('- NCC: ', report['ncc'])
        print('- CF-Score: ', report['cf_score'])

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
