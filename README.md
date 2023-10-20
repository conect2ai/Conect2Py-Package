&nbsp;
&nbsp;
<p align="center">
  <img width="600" src="https://github.com/conect2ai/Conect2Py-Package/assets/56210040/2685de1d-b671-4612-a9e1-d1e0b51d465f" />
</p>

&nbsp;

# Conect2Ai - TAC python package

Conect2Py-Package the name for the Conect2ai Python software package. The package contains the implementation of TAC, an algorithm for data compression using TAC (Tiny Anomaly Compression). The TAC algorithm is based on the concept the data eccentricity and does not require previously established mathematical models or any assumptions about the underlying data distribution.  Additionally, it uses recursive equations, which enables an efficient computation with low computational cost, using little memory and processing power.

Currente version:  ![version](https://img.shields.io/badge/version-0.1.0-blue)

---
#### Dependencies

```bash
Python 3.11, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Ipython
```

---
## Installation

*In progress...*

```bash
pip install tac
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
from tac.models.TAC import TAC
from tac.models.AutoTAC import AutoTAC

# RUN FUNCTIONS
from tac.run.single import (print_run_details)
from tac.run.multiple import (run_multiple_instances, get_optimal_params, display_multirun_optimal_values, run_optimal_combination)

# UTILS FUNCTIONS
from tac.utils.format_save import (create_param_combinations, create_compressor_list, create_eval_df) 
from tac.utils.metrics import (get_compression_report, print_compression_report, calc_statistics)
from tac.utils.plots import (plot_curve_comparison, plot_dist_comparison, plot_multirun_metric_results)

```

### *Running Multiple tests with TAC*
- Setting up the initial variables

```Python
model_name = 'TAC_Voltage_data'

params = {
    'window_size': np.arange(1, 41, 1),
    'm': np.round(np.arange(0.1, 0.8, 0.1), 2),
}

param_combination = create_param_combinations(params)
compressor_list = create_compressor_list(param_combination)
```

- Once you created the list of compressors you can run

```Python
result_df = run_multiple_instances(compressor_list=compressor_list, 
                                param_list=param_combination,
                                series_to_compress=df['voltage'].dropna(), # Example of sensor data
                                cf_score_beta=2
                                )
```

- This function returns a pandas Dataframe containing the results of all compression methods. You can expect something like:

![image](https://github.com/MiguelEuripedes/TACpy/assets/56210040/a7b97cfd-b8b1-424c-8a41-99f8aafb203e)

- You can also check the optimal combination by running the following code:

```Python
display_multirun_optimal_values(result_df=result_df)
```
> Parameter combinations for  MAX CF_SCORE
> 
>            param  reduction_rate  reduction_factor   mse  rmse  nrmse  mae   
>     239  (35, 0.2)            0.96             22.86 37.25  6.10   0.16 1.35  
>     240  (35, 0.3)            0.96             23.30 38.06  6.17   0.16 1.39   
>
>            psnr  ncc  cf_score  
>     239 31.56 0.99      0.98  
>     240 31.47 0.99      0.98  
>Parameter combinations for NEAR  MAX CF_SCORE
>
>
>            param  reduction_rate  reduction_factor   mse  rmse  nrmse  mae   
>     99   (15, 0.2)            0.91             11.61 32.59  5.71   0.15 1.05  \
>     215  (31, 0.6)            0.96             27.59 82.61  9.09   0.23 2.53   
>     157  (23, 0.4)            0.94             17.06 60.59  7.78   0.20 1.59   
>     105  (16, 0.1)            0.92             11.87 32.77  5.72   0.15 1.04   
>     171  (25, 0.4)            0.95             18.24 62.56  7.91   0.20 1.69   
>
>            psnr  ncc  cf_score  
>      99  32.14 0.99      0.97  
>      215 28.10 0.97      0.97  
>      157 29.45 0.98      0.97  
>      105 32.12 0.99      0.97  
>      171 29.31 0.98      0.97  


---

### *Visualize multirun results with a plot*

- By default this plot returns a visualization for the metrics `reduction_rate`, `ncc` and `cf_score`. 
```Python
plot_multirun_metric_results(result_df=result_df)
```
- The result should look like this;

![image](https://github.com/MiguelEuripedes/TACpy/assets/56210040/1c84f28c-7de8-408e-a7e4-5ad5e4e07687)

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
                                                          serie_to_compress=df['voltage'].dropna(),
                                                          model='TAC'
                                                          )
```

- If you want to see the result details use:
```Python
print_run_details(optimal_results_details)
```
> POINTS:
>  - total checked:  50879
>  - total kept:  1114
>  - percentage discaded:  97.81 %
>
> POINT EVALUATION TIMES (ms): 
>  - mean:  0.0021414514134244587
>  - std:  0.046957627024743445
>  - median:  0.0
>  - max:  1.5192031860351562
>  - min:  0.0
>  - total:  108.95490646362305
>
> RUN TIME (ms):
>  - total:  119.3452

---

### *Evaluating the Results*

-  Now, to finish the process of the compression, you should follow the next steps:

**1. Step - Create the evaluation dataframe:**
   
  ```Python
    evaluation_df = create_eval_df(original=df['voltage'].dropna(), flag=points_to_keep)
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
> - Model:  TAC_Voltage_data
> - Optimal Params:  [(35, 0.2), (35, 0.3)]
> - CF-Score Beta:  2
>
> RESULTS 
>
> SAMPLES NUMBER reduction
> - Original length:  50879  samples
> - Reduced length:  1114  samples
> - Samples reduced by a factor of 45.67 times
> - Sample reduction rate: 97.81%
>
> FILE SIZE compression
> - Original size:  544858  Bytes
> - Compressed size:  14165  Bytes
> - file compressed by a factor of 38.47 times
> - file compression rate: 97.4%
>
> METRICS
> - MSE:  41.3406
> - RMSE:  6.4297
> - NRMSE:  0.164
> - MAE:  1.4593
> - PSNR:  31.1085
> - NCC:  0.9865
> - CF-Score:  0.984


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
![image](https://github.com/MiguelEuripedes/TACpy/assets/56210040/ca5cc3d5-841b-4aec-9098-77dc96783369)












# Literature reference

1. Signoretti, G.; Silva, M.; Andrade, P.; Silva, I.; Sisinni, E.; Ferrari, P. "An Evolving TinyML Compression Algorithm for IoT Environments Based on Data Eccentricity". Sensors 2021, 21, 4153. https://doi.org/10.3390/s21124153

2. Medeiros, T.; Amaral, M.; Targino, M; Silva, M.; Silva, I.; Sisinni, E.; Ferrari, P.; "TinyML Custom AI Algorithms for Low-Power IoT Data Compression: A Bridge Monitoring Case Study" - 2023 IEEE International Workshop on Metrology for Industry 4.0 & IoT (MetroInd4.0&IoT), 2023. [10.1109/MetroInd4.0IoT57462.2023.10180152](https://ieeexplore.ieee.org/document/10180152])
