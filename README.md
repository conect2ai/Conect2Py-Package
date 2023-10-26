&nbsp;
&nbsp;
<p align="center">
  <img width="800" src="https://github.com/conect2ai/Conect2Py-Package/assets/56210040/60055d32-77f0-4381-bfc1-c9300eb30920" />
</p> 

&nbsp;


# Conect2Ai - TAC python package

Conect2Py-Package is the name given for the Conect2ai Python software package. The package contains the implementation of TAC, an algorithm for data compression using TAC (Tiny Anomaly Compression). The TAC algorithm is based on the concept the data eccentricity and does not require previously established mathematical models or any assumptions about the underlying data distribution.  Additionally, it uses recursive equations, which enables an efficient computation with low computational cost, using little memory and processing power.

Currente version:  ![version](https://img.shields.io/badge/version-0.1.1-blue)

---
#### Dependencies

```bash
Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Ipython
```

---
## Installation

#### You can download our package from the PyPi repository using the following command:

```bash
pip install conect2py
```

#### If you want to install it locally you download the Wheel distribution from [Build Distribution](https://pypi.org/project/conect2py/0.1.1/#files).

*First navigate to the folder where you downloaded the file and run the following command:*

```bash
pip install conect2py-0.1.1-py3-none-any.whl
```

---

## Example of Use

To begin you can import TACpy using

```Python
# FULL PACKAGE
import tac
```

Or try each of our implemented functionalities

```Python
# MODEL FUNCTIONS
from conect2ai.models.TAC import TAC
from conect2ai.models.AutoTAC import AutoTAC

# RUN FUNCTIONS
from conect2ai.run.single import (print_run_details)
from conect2ai.run.multiple import (run_multiple_instances, get_optimal_params, display_multirun_optimal_values, run_optimal_combination)

# UTILS FUNCTIONS
from conect2ai.utils.format_save import (create_param_combinations, create_compressor_list, create_eval_df) 
from conect2ai.utils.metrics import (get_compression_report, print_compression_report, calc_statistics)
from conect2ai.utils.plots import (plot_curve_comparison, plot_dist_comparison, plot_multirun_metric_results)

```

### *Running Multiple tests with TAC*
- Setting up the initial variables

```Python
model_name = 'TAC_Compression'

params = {
    'window_size': np.arange(2, 30, 1),
    'm': np.round(np.arange(0.1, 2.1, 0.1), 2),
}

param_combination = create_param_combinations(params)
compressor_list = create_compressor_list(param_combination)
```

- Once you created the list of compressors you can run

```Python
result_df = run_multiple_instances(compressor_list=compressor_list, 
                                param_list=param_combination,
                                series_to_compress=dataframe['sensor_data'].dropna(),
                                cf_score_beta=2
                                )
```

- This function returns a pandas Dataframe containing the results of all compression methods. You can expect something like:

|   | param     |	reduction_rate | reduction_factor |	mse	    |  rmse  |	nrmse  |	mae    |	psnr	 | ncc	  | cf_score  |
| - | --------- | -------------- | ---------------- | ------- | ------ | ------- | ------- | ------- | ------ | --------- |
| 0	| (2, 0.1)	|  0.4507    | 1.8204        | 0.0648	| 0.2545 |	0.0609 | 0.0127	 | 39.9824 | 0.9982	| 0.8031    |
| 1	| (2, 0.2)	|  0.4507	   | 1.8204        | 0.0648	| 0.2545 |	0.0609 | 0.0127	 | 39.9823 | 0.9982	| 0.8031    |
| 2	| (2, 0.3)	|  0.4507	   | 1.8204        | 0.0648	| 0.2545 |	0.0609 | 0.0127	 | 39.9823 | 0.9982	| 0.8031    |
| 3	| (2, 0.4)	|  0.4508	   | 1.8209        |	0.0648	| 0.2545 |	0.0609 | 0.0127	 | 39.9824 | 0.9982	| 0.8032    |
| 4	| (2, 0.5)	|  0.4511	   | 1.8217        |	0.0648	| 0.2545 |	0.0609 | 0.0128	 | 39.9823 | 0.9982	| 0.8033    |


- You can also check the optimal combination by running the following code:

```Python
display_multirun_optimal_values(result_df=result_df)
```
> Parameter combinations for  MAX CF_SCORE
> 
>                   param    reduction_rate  reduction_factor     mse      rmse    nrmse  \
>           440  (24, 0.1)          0.9224           12.8919    0.6085  0.7801  0.1867   
>
>                  mae    psnr     ncc    cf_score  
>           440  0.1294  30.254  0.9825    0.9698
> Parameter combinations for NEAR  MAX CF_SCORE
>
>
>             param  reduction_rate    reduction_factor     mse    rmse   nrmse  \
>      521  (28, 0.2)          0.9336           15.0531  1.1504  1.0726  0.2567   
>      364  (20, 0.5)          0.9118           11.3396  0.9458  0.9725  0.2328   
>      262  (15, 0.3)          0.8810            8.4029  0.6337  0.7960  0.1905   
>      363  (20, 0.4)          0.9102           11.1352  0.9084  0.9531  0.2281   
>      543  (29, 0.4)          0.9372           15.9222  1.1474  1.0712  0.2564   
>
>            mae     psnr     ncc     cf_score  
>      521  0.1810  27.4883  0.9666    0.9598  
>      364  0.1431  28.3388  0.9726    0.9598  
>      262  0.0907  30.0780  0.9817    0.9598  
>      363  0.1323  28.5140  0.9737    0.9603  
>      543  0.1925  27.4996  0.9667    0.9607   


---

### *Visualize multirun results with a plot*

- By default this plot returns a visualization for the metrics `reduction_rate`, `ncc` and `cf_score`. 
```Python
plot_multirun_metric_results(result_df=result_df)
```
- The result should look like this;

![image](https://github.com/conect2ai/Conect2Py-Package/assets/56210040/143b1da9-3e45-4ebc-bcc0-2cafd44ec925)



---

### *Running a single complession with the optimal parameter found*

- You don't need to run the visualization and the `display_multirun_optimal_values` in order to get the optimal compressor created, by running the following code it's possible to get the best result: 
```Python
optimal_param_list = get_optimal_params(result_df=result_df)
print("Best compressor param combination: ", optimal_param_list)
```

- With the list of optimal parameter (There is a possibility that multiple compressors are considered the best) run the function below to get get the compression result. 

```Python
points_to_keep, optimal_results_details = run_optimal_combination(optimal_list=optimal_param_list,
                                                          serie_to_compress=dataframe['sensor_data'].dropna(),
                                                          model='TAC'
                                                          )
```

- If you want to see the result details use:
```Python
print_run_details(optimal_results_details)
```
> POINTS:
>  - total checked:  30889
>  - total kept:  1199
>  - percentage discaded:  96.12 %
>
> POINT EVALUATION TIMES (ms): 
>  - mean:  0.003636738161744472
>  - std:  0.15511020000857362
>  - median:  0.0
>  - max:  13.513565063476562
>  - min:  0.0
>  - total:  112.335205078125
>
> RUN TIME (ms):
>  - total:  124.2864

---

### *Evaluating the Results*

-  Now, to finish the process of the compression, you should follow the next steps:

**1. Step - Create the evaluation dataframe:**
   
  ```Python
    evaluation_df = create_eval_df(original=dataframe['sensor_data'].dropna(), flag=points_to_keep)
    evaluation_df.info()
  ```

**2. Step - Evaluate the performance:**
   
```Python
report = get_compression_report(
    original=evaluation_df['original'],
    compressed=evaluation_df['compressed'],
    decompressed=evaluation_df['decompressed'],
    cf_score_beta=2
)

print_compression_report(
    report, 
    model_name=model_name,
    cf_score_beta=2,
    model_params=optimal_param_list
)
```

After that you expect to see something like the following informations:

> RUN INFO 
> - Model:  TAC_Compression
> - Optimal Params:  [(24, 0.1)]
> - CF-Score Beta:  2
>
> RESULTS 
>
> SAMPLES NUMBER reduction
> - Original length:  30889  samples
> - Reduced length:  1199  samples
> - Samples reduced by a factor of 25.76 times
> - Sample reduction rate: 96.12%
>
> FILE SIZE compression
> - Original size:  385549  Bytes
> - Compressed size:  14974  Bytes
> - file compressed by a factor of 25.75 times
> - file compression rate: 96.12%
>
> METRICS
> - MSE:  0.622
> - RMSE:  0.7886
> - NRMSE:  0.1888
> - MAE:  0.1384
> - PSNR:  30.1591
> - NCC:  0.9821
> - CF-Score:  0.9778


**3. Step - Create the model visualizations:**

```Python
# plot the curves comparison (original vs decompressed)
plot_curve_comparison(
    evaluation_df.original,
    evaluation_df.decompressed,
    show=True
)

```

And finally here is a example of the result:

![image](https://github.com/conect2ai/Conect2Py-Package/assets/56210040/978fdeaa-688c-4c8a-90c7-7726eab96302)













# Literature reference

1. Signoretti, G.; Silva, M.; Andrade, P.; Silva, I.; Sisinni, E.; Ferrari, P. "An Evolving TinyML Compression Algorithm for IoT Environments Based on Data Eccentricity". Sensors 2021, 21, 4153. https://doi.org/10.3390/s21124153

2. Medeiros, T.; Amaral, M.; Targino, M; Silva, M.; Silva, I.; Sisinni, E.; Ferrari, P.; "TinyML Custom AI Algorithms for Low-Power IoT Data Compression: A Bridge Monitoring Case Study" - 2023 IEEE International Workshop on Metrology for Industry 4.0 & IoT (MetroInd4.0&IoT), 2023. [10.1109/MetroInd4.0IoT57462.2023.10180152](https://ieeexplore.ieee.org/document/10180152])
