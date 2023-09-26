import numpy as np
import time

# ------------------------------
# RUN SINGLE INSTANCE
# ------------------------------




def run_single_offline(compressor, series_to_compress):
    """
        Returns if the point should be kept or discarded. 
        Besides, returns a dictionary with the run details, which includes the time that took to run the compression in that point.

        Input:
            - compressor: compressor object of TAC of AutoTAC.
            - series_to_compress: series to be compressed (data).
    """
    try:
        # check if compressor is a TAC object or a AutoTAC object
        if (compressor.__class__.__name__ == 'TAC' or compressor.__class__.__name__ == 'AutoTAC'):
            raise ValueError('Compressor must be a TAC object or a AutoTAC object')

        # setup time evaluation
        start_time = time.time()
        # get the coefficients by compressing offline
        coeff = compressor.check_point(series_to_compress)
        # calculate the total time
        end_time = time.time()
        total_time = end_time - start_time
        
        # compile the run details
        run_details = {
            'times': {
                'total': round(total_time*1000, 4),
            }
        }

        return coeff, run_details
    
    except ValueError as err:
        print(f"ValueError: {err}")
        return None, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def run_single_online(compressor, series_to_compress):
    """
        Returns a list of booleans, where True means the point should be kept and False means the point should be discarded. 
        And returns a dictionary with the run details, which includes metrics of time that took to run the compression.
        Run a single compressor online, simulating a datastream with the series_to_compress parameter.    

        Input:
            - compressor: compressor object of TAC of AutoTAC.
            - series_to_compress: series to be compressed (data).
    """
    try:
        # check if compressor is a TAC object or a AutoTAC object
        valid_compressor = ['TAC', 'AutoTAC']
        if compressor.__class__.__name__ not in valid_compressor:
            raise ValueError('Compressor must be a TAC object or a AutoTAC object')

        points_to_keep = []

        # setup time evaluation
        point_eval_times = []
        start_time = time.time()

        # simulate online datastream by iterating through the series
        for index, value in series_to_compress.items():
            # measure the time for each iteration
            iter_start_time = time.time()

            # evaluate the current sensor value
            # if should_keep == True, keep the point, else throw it away
            should_keep = compressor.check_point(value)
            points_to_keep.append(should_keep)

            iter_end_time = time.time()
            iter_total_time = iter_end_time - iter_start_time
            point_eval_times.append(iter_total_time)


        # make the last point o the points_to_keep array equals True for interpolation
        if points_to_keep[0] == False:
            points_to_keep[0] = True

        points_to_keep[-1] = True

        end_time = time.time()
        total_time = end_time - start_time

        # compile the run details
        run_details = {
            'times': {
                'total': round(total_time*1000, 4),
                'iteration_mean': np.mean(point_eval_times) * 1000, 
                'iteration_std': np.std(point_eval_times) * 1000, 
                'iteration_median': np.median(point_eval_times) * 1000, 
                'iteration_max': np.max(point_eval_times) * 1000, 
                'iteration_min': np.min(point_eval_times) * 1000,
                'iteration_total': np.sum(point_eval_times) * 1000, 
                'all_iterations': point_eval_times
            },
            'points': {
                'total_checked': len(points_to_keep),
                'total_kept': np.sum(points_to_keep),
                'total_discarded': len(points_to_keep) - np.sum(points_to_keep),
                'all_kept': points_to_keep
            } 
        }

        return points_to_keep, run_details
    
    except ValueError as err:
        print(f"ValueError: {err}")
        return None, None

    except Exception as e:
        print(f"An error occurred while running 'run_single_onlines': {e}")
        return None, None





# ------------------------------
# RESULTS
# ------------------------------
    




def print_run_details(run_details):
    try:
        percentage_discaded = (run_details['points']['total_discarded']/run_details['points']['total_checked']) * 100
        print('POINTS:')
        print(' - total checked: ', run_details['points']['total_checked'])
        print(' - total kept: ', run_details['points']['total_kept'])
        print(' - percentage discaded: ', round(percentage_discaded, 2), '%')

        print('\nPOINT EVALUATION TIMES (ms): ')
        print(' - mean: ', run_details['times']['iteration_mean'])
        print(' - std: ', run_details['times']['iteration_std'])
        print(' - median: ', run_details['times']['iteration_median'])
        print(' - max: ', run_details['times']['iteration_max'])
        print(' - min: ', run_details['times']['iteration_min'])
        print(' - total: ', run_details['times']['iteration_total'])

        print('\nRUN TIME (ms):')
        print(' - total: ', run_details['times']['total'])
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

