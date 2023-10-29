# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import sys
import datetime

from hierarchical_scheduler import *
from thermal_model import *

from change_dataset import *

def simulate(path,weight,step,pre_step,tmp_in,today,mode):
# def simulate(path,weight,step,pre_step,tmp_in,pre_demand,today,mode):
#path: ファイルのパス
#weight: 目的関数の重み
#step: 計画期間の開始時刻
#pre_step: 予測データの開始時刻
#tmp_in: 現在室温
#today: 実行日
#mode: 運転モード(冷房：0  暖房:1)
    # --- データの読み込み --- #
    weight = float(weight)
    step_path = step.zfill(2)
    step = int(step)
    pre_step = int(pre_step)
    tmp_in = float(tmp_in)
    # pre_demand = float(pre_demand)

    #計画期間のステップ数
    #計画期間÷時間粒度
    period = 96

    #式データの読み込み
    get_column = [0,1,12,24,36,48,60,72,84,96]
    #室温予測に使うデータのカラム
    use_column = (1,2,3)
    #室温予測に使うデータのカラム (室温,消費電力,外気温)
    var = {1:["tmp_in_past","instance.bld_in_tmp_vars"],2:["demand_past","instance.demand"],3:["out_tmp_past","out_tmp"]}
    #変数名
    past_file_name = "past_data.csv"
    # input_file_list = {3:"out_tmp_pre.csv"}

    #式の読み込み
    equation_path = f'{path}/input/using_equation.txt'
    #使える形に成形
    use_equation_list = make_pyomo_equation(equation_path,get_column,use_column,var)

    get_column.reverse()
    use_equation_list.reverse()

    e_path = f'{path}/input/{past_file_name}'
    tmp_in_past = np.array(pd.read_csv(e_path,squeeze=True,header=None,usecols=[1]))[-period:]
    demand_past = np.array(pd.read_csv(e_path,squeeze=True,header=None,usecols=[2]))[-period:]
    out_tmp_past = np.array(pd.read_csv(e_path,squeeze=True,header=None,usecols=[3]))[-period:]

    past_data = np.array(pd.read_csv(e_path,squeeze=True,header=None))[-period:]
    g = open(f'{path}/result/{today}_{step_path}_{weight}/past_data.csv', 'w', newline='',encoding='shift_jis')
    #g = open('out11.csv', 'w', newline='')
    writer = csv.writer(g)
    writer.writerows(past_data)
    g.close()

    # print(tmp_in_past)
    # print(demand_past)
    # print(out_tmp_past)

    # --- データセット読み込み --- #
    # 予測外気温[℃]
    out_tmp = np.array(pd.read_csv(f'{path}/input/temp_pre.csv', squeeze=True, header=None))
    #使用するデータの抜き出し
    if step - pre_step < 0:
        out_tmp = out_tmp[step - pre_step+period:step - pre_step + period+period]
    else:
        out_tmp = out_tmp[step - pre_step:step - pre_step + period]

    # 在室状況　(不在:0, 在室:1)
    occupancy = np.array(pd.read_csv(f'{path}/input/occupancy.csv', squeeze=True, header=None, dtype=int))
    occupancy = occupancy[step : step+ period]

    # #ピーク時刻 (ピークカットoff:0, ピークカットon:1)
    # gamma = np.array(pd.read_csv(f'{path}/input/gamma.csv', squeeze=True, header=None))
    # gamma =gamma[step:step + period]
    # #直前の時刻のgammaの値
    # pre_gamma = gamma[step-1]

    # #ピーク抑制の強さ
    # #gammaの値に応じて消費電力変位に係数を乗じる
    # #これにより、目的関数におけるウェイトを大きくする
    # #gamma = 1 の場合
    # s_on = 1.0
    # #gamma = 0 の場合
    # s_off = 1.0/2

    #目標温度[℃]
    tmp_ref = np.array(pd.read_csv(f'{path}/input/tar_temp.csv', squeeze=True, header=None))
    if step - pre_step < 0:
        tmp_ref =tmp_ref[step - pre_step+96:step - pre_step + period+96]
    else:
        tmp_ref =tmp_ref[step - pre_step:step - pre_step + period]

    #室温の上限、下限[℃]
    tmp_upper = [0 for t in range(period)]
    tmp_lower = [0 for t in range(period)]
    if mode == 0:
        #冷房の場合
        for t in range(period):
            if occupancy[t] == 0:
                #人が不在の場合
                tmp_upper[t] = 40
                tmp_lower[t] = 15
            else:
                #人が在室している場合
                #目標温度からの誤差2℃
                tmp_upper[t] = tmp_ref[t] + 2
                tmp_lower[t] = tmp_ref[t] - 2
    else:
        #暖房の場合
        for t in range(period):
            if occupancy[t] == 0:
                #人が不在の場合
                tmp_upper[t] = 30
                tmp_lower[t] = 0
            else:
                #人が在室している場合
                #目標温度からの誤差2℃
                tmp_upper[t] = tmp_ref[t] + 2
                tmp_lower[t] = tmp_ref[t] - 2

    # --- 空調モデル、建物モデル、シミュレータの準備--- #
    # 建物モデル
    bld_model = firstRC()

    # 空調モデル
    if mode == 0:
        hvac_model = Hitachi_cool()
    else:
        hvac_model = Hitachi_heat()

    #計画期間内に人が在室している場合のシミュレーション
    if sum(occupancy)!=0:
        # 最適化計算を行うクラス
        scheduler = HVAC_Scheduler(bld_model, hvac_model, period)
        # 以下の2つのクラスは目的関数の正規化に使用する
        #
        best_tmp_scheduler = HVAC_Scheduler_sub(bld_model, hvac_model, period)
        worst_tmp_scheduler = HVAC_Scheduler_sub(bld_model, hvac_model, period)

        # ---------------------------- #
        # --- シミュレーション開始　--- #
        # ---------------------------- #

        # --- 目的関数の正規化　--- #
        #消費電力量の最大値
        #この値で、消費電力量の最大変位を正規化
        max_demand = hvac_model.get_hvac_params()['v_upper']/period*24

        # 重みを極端な値に設定して、その結果を正規化に用いる
        # 室温誤差最小、消費電力量最大
        miweight = 0.001
        worst_cost, best_tmp, tmp_prof,demand_prof,switch_prof = best_tmp_scheduler.optimize(use_equation_list,get_column, tmp_ref, tmp_upper, tmp_lower, tmp_in,tmp_in_past, out_tmp,out_tmp_past,demand_past,occupancy, miweight)
        # worst_cost, best_tmp, tmp_prof,demand_prof,switch_prof = best_tmp_scheduler.optimize(use_equation_list,get_column, tmp_ref, tmp_upper, tmp_lower, tmp_in,tmp_in_past, out_tmp,out_tmp_past,demand_past,occupancy,s_on,s_off,gamma,pre_gamma,pre_demand,max_demand, miweight)    
        # 室温誤差最大、消費電力量最小

        mxweight = 0.999
        best_cost, worst_tmp, tmp_prof,demand_prof,switch_prof = worst_tmp_scheduler.optimize(use_equation_list,get_column, tmp_ref, tmp_upper, tmp_lower, tmp_in,tmp_in_past, out_tmp,out_tmp_past,demand_past,occupancy, mxweight)
        # best_cost, worst_tmp, tmp_prof,demand_prof,switch_prof = worst_tmp_scheduler.optimize(use_equation_list,get_column, tmp_ref, tmp_upper, tmp_lower, tmp_in,tmp_in_past, out_tmp,out_tmp_past,demand_past,occupancy,s_on,s_off,gamma,pre_gamma,pre_demand,max_demand, mxweight)
        # 最良消費電力量、最悪消費電力量の差が小さい場合、
        # ここで計算終了

        if abs(worst_cost - best_cost) <= 0.01:
            for i in range(period):
                #demand_profは消費電力量[kWh]のため
                #消費電力[kW]に変換
                demand_prof[i] = demand_prof[i]*period/24
            # 結果のグラフ画像を出力
            plotting(tmp_upper,tmp_lower,tmp_ref,tmp_prof,demand_prof,out_tmp,today,step,weight,period,hvac_model.get_hvac_params()['v_upper'],path)
            return tmp_prof,demand_prof,switch_prof

        # 最適化計算を実行
        tmp_prof, demand_prof,switch_prof = scheduler.optimize(use_equation_list,get_column,best_tmp, worst_tmp, best_cost, worst_cost, tmp_ref, tmp_upper, tmp_lower,tmp_in,tmp_in_past,out_tmp,out_tmp_past,demand_past,occupancy,weight)
        # tmp_prof, demand_prof,switch_prof = scheduler.optimize(use_equation_list,get_column,best_tmp, worst_tmp, best_cost, worst_cost, tmp_ref, tmp_upper, tmp_lower,tmp_in,tmp_in_past, out_tmp,out_tmp_past,demand_past,occupancy,s_on,s_off,gamma,pre_gamma,pre_demand,max_demand, weight)
        for i in range(period):
            #demand_profは消費電力量[kWh]のため
            #消費電力[kW]に変換
            demand_prof[i] = demand_prof[i]*period/24
        # 結果のグラフ画像を出力
        plotting(tmp_upper,tmp_lower,tmp_ref,tmp_prof,demand_prof,out_tmp,today,step,weight,period,hvac_model.get_hvac_params()['v_upper'],path)

    #計画期間内に人が在室していない場合のシミュレーション
    else:
        scheduler = HVAC_Scheduler_night(bld_model, hvac_model, period)
        tmp_prof, demand_prof,switch_prof = scheduler.optimize(tmp_in, out_tmp)
        plotting(tmp_upper,tmp_lower,tmp_ref,tmp_prof,demand_prof,out_tmp,today,step,weight,period,hvac_model.get_hvac_params()['v_upper'],path)

    return tmp_prof,demand_prof,switch_prof

#結果をグラフ画像で出力
def plotting(upper,lower,ref,tmp_prof,demand_prof,out_tmp,today,step,weight,period,max_demand,path):
    xx = [x/period*24+step/period*24 for x in range(period)]
    step_path = str(step).zfill(2)

    fig = plt.figure(figsize=(8.0,6.4))
    ax = fig.add_subplot(2,1,1)
    ln1, = ax.plot(xx,lower,color = "gray",linestyle="--")
    ln2, = ax.plot(xx,upper,color = "gray",linestyle="--",label="temp bound")
    ln3, = ax.plot(xx,ref,color = "k",linestyle="--",label="target temp")

    ln4, = ax.plot(xx,tmp_prof,color="g",label="temp")
    ln5, = ax.plot(xx,out_tmp,color="fuchsia",label="out temp")
    leg1 = ax.legend(bbox_to_anchor=(0.1, 1.02),ncol=4,borderaxespad=1)
    ax.set_ylabel("Temp[°C]")

    bx = fig.add_subplot(2,1,2,ylim=(-0.1,max_demand))

    ln6, = bx.plot(xx,demand_prof,color="g")
    bx.set_ylabel("power consumption[kW]")
    plt.savefig(f"{path}/result/{today}_{step_path}_{weight}/graph.png", bbox_inches='tight')


def main():
    #目的関数の重み
    weight = sys.argv[1]
    #計画期間の開始時刻
    step = sys.argv[2]
    step_path = step.zfill(2)
    #予測データの開始時刻
    pre_step = sys.argv[3]
    #現在室温
    tmp_in = sys.argv[4]
    #前の時刻の消費電力 2023-10-25削除
    # pre_demand = sys.argv[5]

    # path = sys.argv[6]
    path = sys.argv[5]

    #運転モード
    #冷房：0  暖房:1
    mode = 0

    #結果を保存するディレクトリを作成
    today = datetime.date.today()
    os.makedirs(f"{path}/result/{today}_{step_path}_{weight}", exist_ok=True)

    #シミュレーションを行う関数呼び出し
    tmp_prof,demand_prof,switch_prof = simulate(path,weight,step,pre_step,tmp_in,today,mode)
    # tmp_prof,demand_prof,switch_prof = simulate(path,weight,step,pre_step,tmp_in,pre_demand,today,mode)
    #結果を記録
    np.savetxt(f'{path}/result/{today}_{step_path}_{weight}/tmp_prof.csv', tmp_prof,fmt="%f", delimiter=',')
    ##np.savetxt(f'{path}/result/{today}_{step_path}_{weight}/ratio_prof.csv', hvac_ratio,fmt="%f", delimiter=',')
    np.savetxt(f'{path}/result/{today}_{step_path}_{weight}/demand_prof.csv', demand_prof,fmt="%f", delimiter=',')

    result = []
    for t in range(len(tmp_prof)-1):
        result.append([tmp_prof[t+1],demand_prof[t],switch_prof[t]])
    np.savetxt(f'{path}/result.csv',result,delimiter=',')

if __name__ == "__main__":
    main()
