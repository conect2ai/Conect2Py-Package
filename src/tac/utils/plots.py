import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




# ------------------------------
# COMPARISSONS
# ------------------------------




def plot_curve_comparison(reference, processed, start=None, finish=None, show=False, save=False, file_path=None):
    """
    Plot the reference(original) and processed(compressed) curves in the same plot.

    Input:
        - reference: reference data series (original data series)
        - processed: processed data series (compressed data series)
        Optional:
            - start: start index to plot [default: None]
            - finish: finish index to plot [default: None]
            - show: show the plot [default: False]
            - save: save the plot [default: False]
            - file_path: path to save the plot [default: None]
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(15,6))
        ax.clear()

        # define the curve1 and curve2 based on the params
        curve1 = reference if (not start or not finish) else reference[start:finish]
        curve2 = processed if (not start or not finish) else processed[start:finish]
        
        curve1_name = reference.name.capitalize()
        curve2_name = processed.name.capitalize()

        # plot the 2 curves
        ax.plot(curve2, color='b', marker='o', linewidth=2, markersize=5, zorder=5, label=curve2_name)
        ax.plot(curve1, color='r', marker='o', linewidth=2, markersize=5, zorder=1, alpha=0.2, label=curve1_name)

        fig.suptitle(f'{curve1_name} vs {curve2_name}', fontsize=16)
        ax.legend()
        ax.set_ylabel('Sensor Value', fontsize=14)
        ax.set_xlabel('Time', fontsize=14)
        plt.grid(color='k', linestyle=':', linewidth=0.5)
        
        if (save and file_path):
            print('[SAVING]: ', file_path.split('results/')[1])
            plt.savefig(file_path, bbox_inches='tight', facecolor='w', transparent=False)
            if (not show):
                plt.close(fig)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    except Exception as e:
        print('Error plotting curve comparison')
        print(e)
    

def plot_dist_comparison(reference, processed, show=False, save=False, file_path=None):
    """
    Plot the reference(original) and processed(compressed) distributions plot in the same plot.

    Input:
        - reference: reference data series (original data series)
        - processed: processed data series (compressed data series)
        Optional:
            - show: show the plot [default: False]
            - save: save the plot [default: False]
            - file_path: path to save the plot [default: None]
    """
    try:
        fig, ax = plt.subplots(1, 2, figsize=(16,6))

        reference_name = reference.name.capitalize()
        processed_name = processed.name.capitalize()

        sns.distplot(reference, label=reference_name, ax=ax[0])
        sns.distplot(processed, label=processed_name, ax=ax[0])

        sns.ecdfplot(reference, label=reference_name, ax=ax[1], zorder=1)
        sns.ecdfplot(processed, label=processed_name, ax=ax[1], zorder=4)

        for ax_item in ax:
            ax_item.legend()
            ax_item.grid(color='k', linestyle=':', linewidth=0.5)

        fig.suptitle(f'{reference_name} vs {processed_name}', fontsize=16)
        ax[0].set_title('PDF', fontsize=16)
        ax[0].set_ylabel('Density', fontsize=14)
        ax[0].set_xlabel('Data Value', fontsize=14)
        ax[1].set_title('ECDF', fontsize=16)
        ax[1].set_ylabel('Proportion', fontsize=14)
        ax[1].set_xlabel('Sensor Value', fontsize=14)


        if (save and file_path):
            print('[SAVING]: ', file_path.split('results/')[1])
            plt.savefig(file_path, bbox_inches='tight', facecolor='w', transparent=False)
            if (not show):
                plt.close(fig)
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        print('Error plotting dist comparison')
        print(e)





# ------------------------------
# MULTIRUN RESULTS
# ------------------------------




# TODO - try to verify if the metric_list is valid, that is, verify if all the column names are present in the result_df
def plot_multirun_metric_results(result_df, add_trendline=True, metric_list=['reduction_rate', 'ncc', 'cf_score'], show=True, save=False, file_path=None):
    """
    Plot the results of a multirun experiment, showing the evolution of the metrics for each hyperparameter combination.
    Ideal to run after the multirun experiment (run_multiple_instances())
    
    Observations:
        - If the cf_score metric is present, the max value is annotated in the plot.
        - The best visualizations are obtained when the metric_list has 3 metrics or less.

    Input:
        - result_df: dataframe with the results of the multirun experiment
        Optional:
            - add_trendline: add a trendline to the plot [default: True]
            - metric_list: list of metrics to plot [default: ['rate', 'ncc', 'cf_score']]
            - show: show the plot [default: True]
            - save: save the plot [default: False]
            - file_path: path to save the plot [default: None]


    """
    try:
        # verify if the metric_list is valid
        for metric in metric_list:
            if not metric in result_df.columns:
                raise ValueError('Metric_list must contain valid column names from the result_df.')

        fig, ax = plt.subplots(1, len(metric_list), figsize=(18,4))

        for i, metric in enumerate(metric_list):
            x = result_df.index
            y =  result_df[metric]
            params = [f'{param}' for param in result_df.param]
            # add trendline
            if (add_trendline):
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax[i].plot(x,p(x),"r--", alpha=0.5, label='trend')
            # annotate max cf score
            if (metric == 'cf_score'):
                max_y = result_df[metric].max()
                min_y = result_df[metric].min()
                ax[i].set_ylim([min_y - 0.005, max_y + 0.02])
                max_x =  result_df[metric].idxmax()
                ax[i].annotate(f'Max: {y[max_x]}',
                    xy=(max_x, max_y), xycoords='data',
                    xytext=(0.6, 0.98), textcoords='axes fraction',
                    arrowprops=dict(facecolor='grey', edgecolor='grey', shrink=0.1),
                    horizontalalignment='right', verticalalignment='top'
                )
            # make the scatter plot
            ax[i].scatter(x, y, color='b', linewidth=2)
            ax[i].plot(x, y, color='b', linewidth=2, label=metric)
            # add grid and other poperties
            ax[i].grid(color='k', linestyle=':', linewidth=0.5)
            ax[i].legend()
            ax[i].set_title(metric.upper(), fontsize=16)
            ax[i].set_ylabel('Value', fontsize=14)
            ax[i].set_xlabel('Hyperparameters', fontsize=14)
            ax[i].set_xticks(x)
            ax[i].set_xticklabels(params)
            ax[i].locator_params(axis='x', nbins=6)

        if (save and file_path):
            print('[SAVING]: ', file_path.split('results/')[1])
            plt.savefig(file_path, bbox_inches='tight', facecolor='w', transparent=False)
            if (not show):
                plt.close(fig)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    except ValueError as ve:
        print('ValueError plotting multirun metric results')
        print(ve)

    except Exception as e:
        print('Error plotting multirun params results')
        print(e)





# ------------------------------
# INTERNAL MODEL METRICS
# ------------------------------





def plot_model_metrics_dist(model_metrics_dict, metric_list=['m', 'k'], show=False, save=False, file_path=None):
    """
    ADD DESCRIPTION
    """
    try:
        metrics_df = pd.DataFrame(model_metrics_dict)

        fig, ax = plt.subplots(1, len(metric_list), figsize=(16,6))
        fig.suptitle('Model Metrics Ditributions', fontsize=16)

        for i, metric in enumerate(metric_list):
            sns.distplot(metrics_df[metric], label='PDF', ax=ax[i])
            ax[i].set_title('PDF', fontsize=16)
            ax[i].set_ylabel('Density', fontsize=14)
            ax[i].set_xlabel(metric, fontsize=14)

        if (save and file_path):
            print('[SAVING]: ', file_path.split('results/')[1])
            plt.savefig(file_path, bbox_inches='tight', facecolor='w', transparent=False)
            if (not show):
                plt.close(fig)
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        print('Error plotting model metrics dist')
        print(e)