import pandas as pd
import itertools as it
from ..models.TAC import TAC



# ------------------------------
# MODEL EVALUATION
# ------------------------------






def create_eval_df(original, flag):
    """
    Using the original data series and the flag (list of booleans of the same length of the original series), 
    creates a dataframe with the original, compressed and decompressed curves.

    Input:
        - original: original data series
        - flag: list of booleans of the same length of the original series.

    Output:
        - eval_df: dataframe with the original, compressed and decompressed curves.
    
    """
    try:
        # check if flag is a list
        if not isinstance(flag, list):
            raise ValueError('Flag must be a list')
        
        # check if flag is a list of booleans
        if not all(isinstance(item, bool) for item in flag):
            raise ValueError('Flag must be a list of booleans')
        
        # check if flag is a list of booleans of the same length as original
        if len(flag) != len(original):
            raise ValueError('Flag must be a list of booleans of the same length as original')

        # create eval df
        eval_df = pd.DataFrame(original)
        eval_df.columns = ['original']
        
        # decompressed = compressed.reindex(range(len(original))).interpolate(method='linear')
        eval_df['compressed'] = original[flag]
        eval_df['decompressed'] = eval_df.compressed.interpolate(method='values')

        # reset index to avoid inconsistencies
        eval_df.reset_index(inplace=True, drop=True)

        return eval_df
    
    # Handles the value error exception
    except ValueError as err:
        print(f"ValueError: {err}")
        return None
    
    # Handles any other exception
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def create_param_combinations(param_dict):
    """
    Using the param_dict, creates a list of all possible combinations of the parameters. 
    For TAC uses m and window_threshold parameters. And for AutoTAC uses window_threshold parameters.
    
    Input: 
        - param_dict: dictionary with the parameters to combine
    
    Output:
        - param_combinations: list of all possible combinations of the parameters. 
         (Using TAC returns tuples of (window_threshold, m). Using AutoTAC returns tuples of (window_threshold)).

    """
    try:
        # check if param_dict is a dict
        if not isinstance(param_dict, dict):
            raise ValueError('Param_dict must be a dictionary')
        
        # check if param_dict if empty or contains empty params lists
        if not param_dict or any(any(val < 0 for val in values) for values in param_dict.values()):
            raise ValueError('Param_dict must not be empty and/or must not contain negative numbers')

        if len(param_dict.keys()) == 1:
            return list(param_dict.values())[0]

        # create all different param combinations
        param_names = param_dict.keys()
        param_combinations_iter = it.product(*(param_dict[name] for name in param_names))
        param_combinations = list(param_combinations_iter)

        return param_combinations
    
    except ValueError as err:
        print(f"ValueError: {err}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def create_compressor_list(param_combinations):
    """
    Helper function to create a list of compressors ,to be used for TAC, using the param_combinations.
    Interests to use after create_param_combinations() function.

    Input:
        - param_combinations: list of all possible combinations of the parameters.

    Output:
        - compressor_list: list of compressors of TAC.        
    """
    try:
        # check if param_combinations is a list and is not empty
        if not isinstance(param_combinations, list):
            raise ValueError('Param_combinations must be a list')
        
        if not param_combinations:
            raise ValueError('Param_combinations must not be empty')

        compressor_list = [TAC(m=m, window_threshold=window) for (window, m) in param_combinations]

        return compressor_list
    
    except ValueError as err:
        print(f"ValueError: {err}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None





# ------------------------------
# FINAL EXPERIMENT RESULTS
# ------------------------------






def create_compression_report_df(report_dict, save=False, file_path=None):
    """
    Create a dataframe with the results of the compression report. Ideal to use on a dictionary of compression reports.
    Try to use get_compression_report() and save the result on a dictionary. If save and file_path are set as True, the 
    report will be saved on the directory.

    Input:
        - report_dict: dictionary containing compression reports.

    Output:
        - report_df: Pandas DataFrame with the compression results.

    """
    try:
        # check if report_dict is a dict
        if not isinstance(report_dict, dict):
            raise ValueError('Report_dict must be a dictionary')
        

        report_df = pd.DataFrame(report_dict)
        report_df = report_df.T

        report_df = report_df[['file_comp_rate', 'mse',
                            'rmse', 'mae', 'psnr', 'ncc', 'cf_score', 'params']]
        
        report_df['file_comp_rate'] = report_df['file_comp_rate']*100

        report_df.columns = [
            'File Compression Rate (%)', 'MSE', 'RMSE', 'MAE', 'PSNR', 'NCC', 'CF-Score', 'Parameters']
        
        report_df = report_df.reindex(report_dict.keys())

        if (save and file_path):
            print('[SAVING]: ', file_path.split('results/')[1])
            report_df.to_csv(file_path)

        return report_df
    
    except ValueError as err:
        print(f"ValueError: {err}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None




def create_stats_report_df(report_dict, save=False, file_path=None):
    """
    Create a dataframe with the statistics results of the compression dictionary report.
    Ideal to run after concluding the experiments. Once that is done, create a dictionary of stats reports to be used as input, ideally
    use the name of the model to discern between runs.
    Use calc_statistics() and save the results on a dictionary.

    Input:
        - report_dict: Dictionary containing the stats reports.

    Output:
        - report_df: Pandas DataFrame with the statistics compression results.

    """
    try:
        # check if report_dict is a dict
        if not isinstance(report_dict, dict):
            raise ValueError('Report_dict must be a dictionary')

        # create a multi index df
        report_df = pd.concat(report_dict.values(), keys=report_dict.keys())
        report_df = round(report_df, 2)

        if (save and file_path):
            print('[SAVING]: ', file_path.split('results/')[1])
            report_df.to_csv(file_path)

        return report_df
    
    except ValueError as err:
        print(f"ValueError: {err}")
        return None
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
