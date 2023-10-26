import numpy as np
import pandas as pd

import IPython.display as display

from ..utils.format_save import create_eval_df
from ..utils.metrics import get_compression_report
from ..models.TAC import TAC
from ..models.AutoTAC import AutoTAC

from .single import run_single_online, run_single_offline





# ------------------------------
# RUN MULTIPLE INSTANCES 
# ------------------------------





def run_multiple_instances(compressor_list, param_list, series_to_compress, cf_score_beta=1, compressor_type='online'):
    """
    Run multiple instances of a compressor and return a dataframe with the results.
    The compressor_list and param_list must have the same length, doe to the fact that each compressor will be run with a different set of parameters, created using the param combinations.
    
    Input:
        - compressor_list: list of compressors of TAC.
        - param_list: list of parameters to be used on each compressor.
        - series_to_compress: series to be compressed (data).
        Optional:
            - cf_score_beta: beta value to be used on the cf_score calculation [default=1].
            - compressor_type: type of compressor to be used, either 'online' or 'offline' [default='online'].

    Output:
        - result_df: dataframe with the results of the compression report.
            .: In this dataframe the "rate" columns are the sample reduction rate and the "factor" columns are the sample reduction factor. To get the size reduction and futher information check the get_compression_report() function.
    
    """
    
    try:
        # check if compressor type is either 'online' or 'offline'
        if not (compressor_type == 'online' or compressor_type == 'offline'):
            raise ValueError('Compressor_type must be either Online or Offline')
        
        # check if the compressor list, param list or series to compress are empty
        if (len(compressor_list) == 0 or len(param_list) == 0 or len(series_to_compress) == 0):
            raise ValueError('Parameters pompressor_list, param_list or series_to_compress should not be empty') 
    


        result = {}
        
        # for each compressor run one time

        for index, compressor in enumerate(compressor_list):
            #print(index)
            eval_df = None
            if (compressor_type == 'online'):
                # run the compression online
                points_to_keep, _ = run_single_online(compressor, series_to_compress)
                # create an eval_df
                eval_df = create_eval_df(series_to_compress, points_to_keep)
            elif (compressor_type == 'offline'):
                # run the compression  offline
                coefficients, _ = run_single_offline(compressor, series_to_compress)

                # decompress the series
                decompressed = compressor.decompress(coefficients)

                # create an eval_df
                eval_df = pd.DataFrame({
                    'original': series_to_compress, 
                    'compressed': coefficients.replace(0, np.nan),
                    'decompressed': decompressed
                })
            else:
                return None

            # calculate compression report metrics
            report = get_compression_report(
                eval_df.original, 
                eval_df.compressed, 
                eval_df.decompressed,
                cf_score_beta=cf_score_beta
            )

            # append the result
            result[index] = {
                'param': param_list[index],
                'reduction_rate': report['sample_reduction_rate'],
                'reduction_factor': report['sample_reduction_factor'],
                'mse': report['mse'],
                'rmse': report['rmse'],
                'nrmse': report['nrmse'],
                'mae': report['mae'],
                'psnr': report['psnr'],
                'ncc': report['ncc'],
                'cf_score': report['cf_score'],
            }

        # create a dataframe from the result
        result_df = pd.DataFrame.from_dict(result, orient='index')

        return result_df
    
    except ValueError as err:
        err_type = type(err).__name__
        mensage = str(err)
        print(f"ValueError occurred while running 'run_multiple_instances': {mensage} - {err_type}")
        return None
    
    except Exception as e:
        print(f"An error occurred while running 'run_multiple_instances': {e}")





# ------------------------------
# RESULTS
# ------------------------------




def get_optimal_params(result_df, metric='cf_score', comp='max'):
    """
        From the result dataframe of running multiple instances of TAC, get the optimal parameters from a result dataframe.
        Ideally, it should be used after running the run_multiple_instances() function.
        For this matter we choose the optimal parameters to be the ones that have the highest cf_score.
        

        Input:
            - result_df: dataframe with the results of the compression report.

        Output:
            - optimal_params: a list of optimal parameters. Base on the highest cf_score.
    """
    try:
        # check if result_df is a dataframe
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError('Result_df must be a dataframe')
        
        # check if result_df has informed metric
        if not metric in result_df.columns:
            raise ValueError('Result_df must have the informed metric column:', metric)
        
        # check if comp is either 'max' or 'min'
        if not (comp == 'max' or comp == 'min'):
            raise ValueError('Comp must be either Max or Min')
        
        # get the optimal parameters
        if comp == 'max':
            #get the column acording to the metric
            metric_score = result_df[result_df[metric] == result_df[metric].max()]
        elif comp == 'min':
            #get the column acording to the metric
            metric_score = result_df[result_df[metric] == result_df[metric].min()]

        return list(metric_score.param.values)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    




def run_optimal_combination(optimal_list, serie_to_compress, model='TAC'):
    optimal_dict = {
        'window': optimal_list[0][0],
        'm': optimal_list[0][1],
    }

    if model == 'TAC':
        compressor = AutoTAC(m=optimal_dict['m'], window_size=optimal_dict['window'], save_metrics=False)
    elif model == 'AutoTAC':
        compressor = AutoTAC(window_size=optimal_dict['window'], save_metrics=False)

    points_to_keep, run_details = run_single_online(compressor, serie_to_compress)

    return points_to_keep, run_details






def display_multirun_optimal_values(result_df, metric='cf_score', comparator='max'):
    """
        From the result dataframe of running multiple instances of TAC, print the line with the optimal parameters and the line with the parameters that are close (0.01) to the optimal.
    """
    try:
        # check if result_df is a dataframe
        if not isinstance(result_df, pd.DataFrame):
            raise ValueError('Result_df must be a dataframe')
        
        # check if result_df has informed metric
        if not metric in result_df.columns:
            raise ValueError('Result_df must have the informed metric column:', metric)
        
        # check if comp is either 'max' or 'min'
        if not (comparator == 'max' or comparator == 'min'):
            raise ValueError('Comp parameter must be either Max or Min')

        if comparator == 'max':
            print('Parameter combinations for ', comparator.upper(), metric.upper()) 
            print('\n')

            max_metric_lines = result_df[result_df[metric] == result_df[metric].max()]
            max_metric_score = max_metric_lines[metric].values[0]        
            print(max_metric_lines)

            print('Parameter combinations for NEAR ', comparator.upper(), metric.upper())
            print('\n')
            
            # find the values that are close to, but not exactly the max
            all_metrics_scores = list(result_df[metric])
            is_close = np.isclose(all_metrics_scores, max_metric_score, atol=0.01)

            print(result_df[is_close & (result_df[metric] != max_metric_score)].sort_values(by=[metric]).head())
        elif comparator == 'min':
            print('Parameter combinations for ', comparator.upper(), metric.upper()) 
            print('\n')
            min_metric_lines = result_df[result_df[metric] == result_df[metric].min()]
            min_metric_score = min_metric_lines[metric].values[0]
            
            display(min_metric_lines)

            print('Parameter combinations for NEAR ', comparator.upper(), metric.upper())
            # find the values that are close to, but not exactly the max
            all_metrics_scores = list(result_df[metric])
            is_close = np.isclose(all_metrics_scores, min_metric_score, atol=0.01)

            display(result_df[is_close & (result_df[metric] != min_metric_score)].sort_values(by=[metric]).head())

    except ValueError as err:
        print(f"ValueError: {err}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    




