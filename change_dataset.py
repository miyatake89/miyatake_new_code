import numpy as np
import pandas as pd
import csv

def make_pyomo_equation(equation_path,get_column,use_column,var):
#equation_path:式が格納されているパス
#get_column: 室温予測に使うデータのカラム
#use_column: 室温予測に使うデータのカラム (室温,消費電力,外気温)
#var: 変数の名前  
    #室温予測に使うデータのカラム
    column_len = len(get_column)
    #室温予測に使うデータのカラム (室温,消費電力,外気温)
    use_var_len = len(use_column)
    #変数の数
    all_var_num = column_len*use_var_len
    var_reverse = dict(reversed(var.items()))
    gc_reverse = get_column
    gc_reverse.reverse()
    print(var_reverse)
    with open(equation_path,mode="r") as f:
        use_equation = f.readline()
    use_equation = use_equation.split("|")[2]
    top = "instance.bld_in_tmp_vars[t+1] =="
    depend_var_list = list(var)
    use_equation_list = dict()

    eq_list = [top + use_equation for j in get_column]
    for s in gc_reverse:
        for t,u in var_reverse.items():
            '''
            print(t)
            print(u)
            print(all_var_num-1)
            '''
            all_var_num -=1
            x_var = "x" + str(all_var_num)
            #print(x_var)

            ind = 0
            for v,w in zip(get_column,eq_list):
                '''
                print(f"{s=}")
                print(f"{v=}")
                print(f"{w=}")
                '''
                if v <= s:
                    rep_str = u[0] + "[96 - " + str(s) + "+t]"
                else:
                    rep_str = u[1] + "[t-" + str(s) + "]"
                eq_list[ind] = eq_list[ind].replace(x_var,rep_str)
                ind += 1
    print(eq_list)
    print(len(eq_list))
    return eq_list

#not use
def make_pyomo_equation_row(equation,get_column,use_column,var):
    column_len = len(get_column)
    use_var_len = len(use_column)
    all_var_num = column_len*use_var_len
    var_reverse = dict(reversed(var.items()))
    gc_reverse = get_column
    gc_reverse.reverse()
    print(var_reverse)
    use_equation = equation
    use_equation = use_equation.split("|")[2]
    top = "instance.bld_in_tmp_vars[t+1] =="
    depend_var_list = list(var)
    use_equation_list = dict()

    eq_list = [top + use_equation for j in get_column]
    for s in gc_reverse:
        for t,u in var_reverse.items():
            '''
            print(t)
            print(u)
            print(all_var_num-1)
            '''
            all_var_num -=1
            x_var = "x" + str(all_var_num)
            #print(x_var)

            ind = 0
            for v,w in zip(get_column,eq_list):
                '''
                print(f"{s=}")
                print(f"{v=}")
                print(f"{w=}")
                '''
                if v <= s:
                    rep_str = u[0] + "[24 - " + str(s) + "+t]"
                else:
                    rep_str = u[1] + "[t-" + str(s) + "]"
                eq_list[ind] = eq_list[ind].replace(x_var,rep_str)
                ind += 1

    return eq_list

#not use
def make_input_postdata(get_column,use_column,depend_var):
    first_n = get_column[0]
    top = max(get_column)
    input_data = np.array(pd.read_csv(f"./input/AC_data_input_minoh.csv", squeeze=True,header=None))
    if first_n == 0:
        base = input_data[top-first_n:,use_column]
    else:
        base = input_data[top-first_n:-first_n,use_column]

    for t in get_column:
        if t != first_n:
            base_next = input_data[top-t:-t,use_column]
            base = np.hstack([base,base_next])

    base = base[-24:]
    g = open(f"./input/past_data.csv", 'w', newline='',encoding='shift_jis')
    #g = open('out11.csv', 'w', newline='')
    writer = csv.writer(g)
    writer.writerows(base)
    g.close()

    return base

if __name__ == "__main__":
    #1:power
    #4:tmp_in
    #6:demand
    #7:tmp_out
    #8:solar
    get_column = [0,1,12,24,36,48,60,72,84,96]
    use_column = (1,4,6,7,8)
    var = {1:["power","instance.power"],4:["tmp_in","instance.tmp_in"],6:["demand","instance.demand"],7:["out_tmp","pre_out_tmp"],8:["solar","pre_solar"]}
    file_name = "past_data.csv"
    input_data_directory = "input"
    input_file_list = {7:"out_tmp_osaka.csv",8:"solar_pre.csv"}
    use_equation_list = make_pyomo_equation(get_column,use_column,var)
    #base = make_input_postdata(get_column,use_column,var)

    g = open(f".result.csv", 'w', newline='',encoding='shift_jis')
    #g = open('out11.csv', 'w', newline='')
    writer = csv.writer(g)
    for n in use_equation_list:
        writer.writerow([n])
    g.close()
