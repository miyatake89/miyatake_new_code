# optimize code in openblocks
Author: Shunsaku Miyatake Date: 2023-10-25  
This is the python code for the optimization framework.

## 1. Requirements
CPLEX (MIP solver; free version for academic)  
Ipopt (NLP solver; free download from Coin-or projects)  
Python3.x  
Python packages :  
Numpy, Pandas, Pyomo, Matplotlib
## 2. execution command
Execute the following command  
<!-- `python main.py weight step pre_step tmp_in pre_demand`  
ex : `python main.py 0.1 0 0 15 0`   -->
`python main.py weight step pre_step tmp_in path`  
ex : `python main.py 0.1 0 0 15`  
If the simulation results are output to the "result/YYYYY-MM-dd_00_0.1" directory, and image and csv files are output, there is no problem.  
Examples of test run results are stored in the RESULT folder.

## 3. main.py
### 3.1 function list
|name|meaning|input variable name|input meaning|output variable name|output meaning|
|---|---|---|---|---|---|
|simulate|Calculation for normalization. After that, the calculation for optimization is performed.|path,  weight,  step,  pre_step,  tmp_in,  today,  mode|file directory,  Objective function weights,  Start time of the planning period,  Start time of forecast data,  Current room temperature,  Today's Date,  Cooling or Heating|tmp_prof,  demand_prof,  switch_prof|"Room temperature transition,  demand transition,  on/off transition"|
|plotting|Create a graph with the results of the optimization calculations.|upper,  lower,  ref,  tmp_prof,  demand_prof,  out_tmp,  today,  step,  weight,  period,  max_demand,  path|Upper temperature,  lower temperature,  Target temperature, Temperature Trend Prediction,  Power consumption transition forecast,  outside temperature,  Today's Date,  Start time of the planning period,  Objective function weights,  planning period(96),  Maximum displacement of power consumption,  file directory|graph.png|result graph|
### 3.2 variables list
|name|meaning|type|
|:----|:----|:----|
|weight|Objective function weights|float|
|step|Start time of the planning period|int|
|pre_step|Start time of forecast data|int|
|tmp_in|Current room temperature|float|
|path|file directory|str|
|mode|cooling：0  heating : 1|int|
|period|Number of steps in planning period (96)|int|
|get_column|Which time data to use(now,15min ago, 1hour ago...)|array|
|use_column|What data to use
(room temperature, outside temperature, Power consumption)|array|
|var|variable name|dictionary|
|past_file_name|File name of past data|str|
|equation_path|Directory containing expressions|str|
|use_equation_list|Room temperature prediction equation in a form that can be used in pyomo|array|
|e_path|file directory ({path}/input/{past_file_name})|str|
|past_data|Array of the above three combined|array|
|out_tmp|Predicted outside temperature|array|
|occupancy|occupancy (0 or 1)|array|
|tmp_ref|Target temperature|array|
|tmp_upper|Upper limit of room temperature|array|
|tmp_lower|Lower limit of room temperature|array|
|bld_model|Building Model|class|
|hvac_model|HVAC Model|class|
|scheduler|Class that performs optimization calculations|class|
|best_tmp_scheduler|Class for normalization|class|
|worst_tmp_scheduler|Class for normalization|class|
| max_demand|Maximum power consumption|float|
|worst_cost|Power consumption at minimum weight|float|
| best_tmp|Room temperature error at minimum weight|float|
|best_cost|Power consumption at maximum weight|float|
| worst_tmp|Room temperature error at maximum weight|float|
|tmp_prof|Room temperature transition|array|
|demand_prof|demand transition|array|
|switch_prof|on/off transition|array|
## 4. hierarchical_scheduler.py
### 4.1 function list
| name          | meaning                                                     | class name           | input variable name                                                                                                                                                                                             | input meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                | output variable name                                                                      | output meaning                                                                                                                                                         |
|---------------|-------------------------------------------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| optimize      | Calculate optimization in the absence   of people.          | HVAC_Scheduler_night | self,      bld_tmp_ini,      out_tmp                                                                                                                                                                            | self,      Initial room temperature,      outside temperature                                                                                                                                                                                                                                                                                                                                                                                                                | tmp_prof,      hvac_ratio,      demand_prof                                               | Target   temperature,      Power consumption ratio,      demand transition                                                                                             |
| optimize      | Calculate the   optimization       when people are present.	 | HVAC_Scheduler       | self,      ueq_list,       get_column,       tmp_ref,       tmp_upper,       tmp_lower,        bld_tmp_ini,      out_tmp,      occupancy,      weight                                                           | self,      result make_pyomo_equation,      What data to use      (room temperature, outside temperature, Power consumption),      Target temperature,      Upper temperature,      Lower limit temperature,      Room temperature just before,      outside temperature,      occupancy,      Objective function weights                                                                                                                                                    | tmp_prof,      hvac_ratio,      demand_prof                                               | Target   temperature,      Power consumption ratio,      demand transition                                                                                             |
| setup_abstmdl | Definition of Variables                                     | HVAC_Scheduler       | self                                                                                                                                                                                                            | 　                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 　                                                                                        | 　                                                                                                                                                                     |
| total_rule    | objective function                                          | HVAC_Scheduler       | instance                                                                                                                                                                                                        | 　                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | objective                                                                                 | 　                                                                                                                                                                     |
| optimize      | Calculate the   optimization       for normalization	        | HVAC_Scheduler_sub   | self,      ueq_list,      get_column,      best_tmp,      worst_tmp,      best_cost,      worst_cost,      tmp_ref,      tmp_upper,      tmp_lower,      bld_tmp_ini,      out_tmp,      occupancy,      weight | self,      result make_pyomo_equation,      What data to use      (room temperature, outside temperature, Power consumption),      Weight best room temperature,      Weight worst room temperature,      Weight worst power consumption,      Weight highest power consumption,      Target temperature,      Upper temperature,      Lower limit temperature,      Room temperature just before,      outside temperature,      occupancy,      Objective function weights | cost_prof,       tmp_prof,       tmp_prof_value,      demand_prof_value,      switch_prof | Total power   consumption,      Sum of squares of error from target room temperature,      Room temperature transition,      demand transition,      on/off transition |
| setup_abstmdl | Definition of Variables                                     | HVAC_Scheduler_sub   | self                                                                                                                                                                                                            | 　                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 　                                                                                        | 　                                                                                                                                                                     |
| total_rule    | objective function                                          | HVAC_Scheduler_sub   | instance                                                                                                                                                                                                        | 　                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | objective                                                                                 | 　                                                                                                                                                                     |
### 4.2 variables list
|name|meaning|type|
|:----|:----|:----|
|th|Air conditioning load factor|float|
|t_period|Number of steps in planning period (96)|int|
|step_length|Time granularity [sec].|int|
|bld_params|Building Model parameter|dictionary|
|hvac_params|HVAC Model parameter|dictionary|
|tmp_prof|Room temperature transition|array|
|hvac_ratio|Air conditioning load factor|array|
|demand_prof|demand transition|array|
|tmp_now|Current room temperature|float|
|next_tmp|next room temperature|float|
|solver|Solver to be used|str|
|mdl|Modeling optimization problems with pyomo| |
|IDX|index| |
|demand_ratio|Power consumption ratio|NonNegativeReals|
|demand|Power consumption|NonNegativeReals|
|purchased_vars|Purchased Electricity|NonNegativeReals|
|bld_in_tmp_vars|room temperature|Reals|
|demand_disp_max|Displacement of power consumption|Reals|
|switch|on/off |Binary|
|tmp_slack_vars|Temperature Slack Parameter|Reals|
|pe_coeff|Penalty Size|int|
|cost_prof|Total power consumption|float|
|tmp_prof|Sum of squares of error from target room temperature|float|
## 5. change_dataset.py
### 5.1 function list
| name                | meaning                                                                                 | file name         | input variable name                                       | input meaning                                                                                                                                                                                      | output variable name | output meaning                                                                   |
|---------------------|-----------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|----------------------------------------------------------------------------------|
| make_pyomo_equation | return Room   temperature prediction equation in a form that can be used in pyomo       | change_dataset.py | equation_path,      get_column,      use_column,      var | file   directory,      Which time data to use       (now,15min ago, 1hour ago...),      What data to use      (room temperature, outside temperature, Power consumption),      variable name       | eq_list              | Room temperature   prediction equation in a form that can be used in pyomo       |
### 5.2 variables list
|name|meaning|type|
|:----|:----|:----|
|get_column|Which time data to use(now,15min ago, 1hour ago...)|array|
|use_column|What data to use
(room temperature, outside temperature, Power consumption)|array|
|file_name|output file name|str|
|input_data_directory|input file name|str|
|input_file_list|What data to use
(outside temperature, solar)|array|
|use_equation_list|result make_pyomo_equation|array|
|var|variable name|dictionary|
|column_len|Column Size|int|
|use_var_len|Number of variable types|int|
|all_var_num|Number of variables|int|
|var_reverse|variable name reverse|dictionary|
|gc_reverse|Which time data to use(now,15min ago, 1hour ago...) reverse|array|
|use_equation|Room temperature prediction formula|str|
## 6. thermal_model.py
### 6.1 function list
|name|meaning|class name|
|:----|:----|:----|
|get_building_params|Return building parameters|firstRC|
|get_hvac_params|Return HVAC parameters|Hitachi_cool|
|get_hvac_params|Return HVAC parameters|Hitachi_heat|
|get_hvac_params|Return HVAC Capability|PiecewiseCOP_cool|
|get_hvac_params|Return HVAC Capability|PiecewiseCOP_heat|
### 6.2 variables list
|name|meaning|type|
|:----|:----|:----|
|bld_type|Building Type|str|
|R|R(0.00389971615331053)|float|
|C|C(7879964.66865623)|float|
|tau|R*C|float|
|params|parameter|dictionary|
|hvac_type|HVAC type|str|
|tmp_upper|Upper limit of room temperature|int|
|tmp_lower|Lower limit of room temperature|int|
| v_upper|Rated power consumption|float|
|xdata,ydata|Piecewise linear function data|dataframe|
|p_upper|Rated output|float|
