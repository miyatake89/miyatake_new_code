#update Shunsaku Miyatake Date: 2023-10-25
#version : 1.0
#Add comments about functions and variables.
#Remove unnecessary variables such as s_on.

# -*- coding: utf-8 -*-
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints

# --- Create the cplex solver plugin using the ASL interface --- #
stream_solver = True # True prints solver output to screen
keepfiles = False    # True prints intermediate file names (.nl,.sol,...)

#空調負荷率の下限
#空調負荷率は0~thの値を取らない
th = 0.06

#人が在室しない場合のシミュレータ
class HVAC_Scheduler_night:
    def __init__(self, bld_model, hvac_model, t_period, step_length=900, **kwargs):
    #bld_model: 建物モデル
    #hvac_model: 空調モデル
    #t_period: 計画期間のステップ数
    #step_length: 時間粒度[秒]
        #計画期間のステップ数
        self.t_period = t_period
        #時間粒度[秒]
        self.step_length = step_length
        #建物モデル
        self.bld_params = bld_model.get_building_params()
        #空調モデル
        self.hvac_params = hvac_model.get_hvac_params()

    def optimize(self, bld_tmp_ini, out_tmp):
    # def optimize(self, tmp_ref, tmp_upper, tmp_lower, bld_tmp_ini, out_tmp, occupancy,weight):
    # 2023-10-25変更
    #bld_tmp_ini: 室内の初期温度[℃]
    #out_tmp: 外気温度[℃]
        #計画期間のステップ数
        t_period = self.t_period
        #室温変化を格納するリスト
        tmp_prof = [bld_tmp_ini]
        #空調負荷率を格納するリスト
        hvac_ratio = []
        #消費電力を格納するリスト
        demand_prof = []
        #現在室温
        tmp_now = bld_tmp_ini
        #次の時刻の室温
        next_tmp = -1

        #消費電力、出力が全期間で0の場合の室温変化を求める
        for t in range(t_period):
            hvac_ratio.append(0)
            demand_prof.append(0)
            if t != t_period:
                next_tmp = (1-(self.step_length/self.bld_params['tau']))*tmp_now + \
                    self.step_length/self.bld_params['tau']*(out_tmp[t])
                tmp_prof.append(next_tmp)
                tmp_now = next_tmp
        tmp_prof = tmp_prof[:t_period]
        return tmp_prof, hvac_ratio, demand_prof

#人が在室している場合のシミュレータ
class HVAC_Scheduler:
    def __init__(self, bld_model, hvac_model, t_period, step_length=900, **kwargs):
    #bld_model: 建物モデル
    #hvac_model: 空調モデル
    #t_period: 計画期間のステップ数
    #step_length: 時間粒度[秒]
        #計画期間のステップ数
        self.t_period = t_period
        #時間粒度[秒]
        self.step_length = step_length
        #pyomoによる、最適化問題のモデル化
        self.mdl = AbstractModel()
        #建物モデル
        self.bld_params = bld_model.get_building_params()
        #空調モデル
        self.hvac_params = hvac_model.get_hvac_params()
        #使用するソルバー
        solver = "cplex"
        solver_io = "nl"
        self.opt = SolverFactory(solver)
        #self.opt.options['optimalitytarget'] = 3
        #self.opt.options['optimalitytarget'] = 3
        #self.opt = SolverFactory(solver, solver_io=solver_io)

        #最適化計算に時間制限を設ける
        self.opt.options['timelimit'] = 60*3

        if self.opt is None:
            print("")
            print("ERROR: Unable to create solver plugin for %s "
                  "using the %s interface" % (solver, solver_io))
            print("")
            exit(1)
        self.setup_abstmdl()

    def setup_abstmdl(self):
        mdl = self.mdl
        t_period = self.t_period

        # -- 変数の定義 -- #
        mdl.IDX = Set(initialize=range(t_period))
        #消費電力比
        mdl.demand_ratio = Var(mdl.IDX,bounds=(0,1), within=NonNegativeReals,initialize=0)
        #消費電力[kW]
        mdl.demand = Var(mdl.IDX, within=NonNegativeReals,initialize=0)
        #購入電力量[kWh]
        mdl.purchased_vars = Var(mdl.IDX, within=NonNegativeReals,initialize=0)
        #室温[℃]
        mdl.bld_in_tmp_vars = Var(range(t_period+1), within=Reals,initialize=0)
        #消費電力量の変位[kWh]
        mdl.demand_disp_max = Var(within=Reals,initialize=0)
        #空調のON/OFF

        #switchが 0の場合、off　1の場合、on
        mdl.switch = Var(mdl.IDX, within=Binary,initialize=0)


        mdl.tmp_slack_vars = Var(mdl.IDX, bounds=(0,5), within=Reals,initialize=0)
        mdl.pe_coeff = 1000000

    def optimize(self, ueq_list, get_column, best_tmp, worst_tmp, best_cost, worst_cost, tmp_ref, tmp_upper, tmp_lower,  bld_tmp_ini,tmp_in_past,out_tmp,out_tmp_past,demand_past,occupancy, weight):
    # def optimize(self, ueq_list, get_column, best_tmp, worst_tmp, best_cost, worst_cost, tmp_ref, tmp_upper, tmp_lower,  bld_tmp_ini,tmp_in_past, out_tmp,out_tmp_past,demand_past, occupancy, s_on,s_off,gamma,pre_gamma,pre_demand,max_disp,weight):
    # 2023-10-25変更
    #ueq_list: 室温予測式
    #get_column: 室温予測に使うデータのカラム
    #best_tmp: 目標室温との誤差の二乗の総和の最小値
    #worst_tmp: 目標室温との誤差の二乗の総和の最大値
    #best_cost: 消費電力量の総和の最小値
    #worst_cost: 消費電力量の総和の最大値
    #tmp_ref: 目標室温[℃]
    #tmp_upper: 室温の上限[℃]
    #tmp_lower: 室温の下限[℃]
    #bld_tmp_ini: 室内の初期温度[℃]
    #out_tmp: 外気温度[℃]
    #occupancy: 在室状況
    #weight: 目的関数の重み

        instance = self.mdl.create_instance()
        #計画期間における室温の初期値を設定
        instance.first_tmp = Constraint(expr = instance.bld_in_tmp_vars[0] == bld_tmp_ini)
        #室温の上限と下限を設定
        instance.tmp_bound = ConstraintList()
        for t in instance.IDX:
            instance.tmp_bound.add(instance.bld_in_tmp_vars[t] <= tmp_upper[t] + instance.tmp_slack_vars[t])
            instance.tmp_bound.add(instance.bld_in_tmp_vars[t] >= tmp_lower[t] - instance.tmp_slack_vars[t])

        '''
        #空調モデルの区分線形関数を設定
        for t in instance.IDX:
            #外気温が想定範囲から超えている場合
            #収まる値に変更
            ot = int(out_tmp[t])
            if ot < self.hvac_params['tmp_lower']:
                ot = self.hvac_params['tmp_lower']
            if ot > self.hvac_params['tmp_upper']:
                ot = self.hvac_params['tmp_upper']
            #区分線形関数を設定
            #動的に宣言しているため、exec関数を使用
            exec(f"instance.con_{t} = Piecewise(instance.demand_ratio[{t}],instance.power_ratio[{t}],pw_pts=self.hvac_params['xdata'][{ot-self.hvac_params['tmp_lower']}],pw_constr_type='EQ',f_rule=self.hvac_params['ydata'][{ot-self.hvac_params['tmp_lower']}],pw_repn='SOS2')")
        '''

        #消費電力に関する制約リスト
        instance.voltage_con = ConstraintList()
        for t in instance.IDX:
            #外気温が想定範囲から超えている場合
            #収まる値に変更
            ot = int(out_tmp[t])
            if ot < self.hvac_params['tmp_lower']:
                ot = self.hvac_params['tmp_lower']
            if ot > self.hvac_params['tmp_upper']:
                ot = self.hvac_params['tmp_upper']
            #購入電力量と消費電力比の関係
            instance.voltage_con.add(instance.purchased_vars[t] == instance.demand[t]*(self.step_length/3600))
            instance.voltage_con.add(instance.demand[t] == self.hvac_params['v_upper']*instance.demand_ratio[t])

            #switchが0の場合、power_ratioも0になり
            #switchが1の場合、power_ratioはth以上になる
            instance.voltage_con.add(instance.demand_ratio[t] >= th*instance.switch[t])
            instance.voltage_con.add(instance.demand_ratio[t] <= instance.switch[t])

        #室温変化の制約式
        instance.tmp_control = ConstraintList()
        for t in instance.IDX:
            mae = -1
            for gc,eqa in zip(get_column,ueq_list):
                if t >= mae and t < gc:
                    print(f"{t=}")
                    print(eqa)
                    exec(f"instance.tmp_control.add({eqa})")
                    break
                mae = gc
        
        # ピーク項の制約 2023-10-25削除
        # #消費電力変位の制約式
        # instance.displacement_bound = ConstraintList()
        # #変数gammaが1の場合、消費電力変位に係数を乗じる
        # #これによって、目的関数におけるウェイトを大きくする
        # instance.displacement_bound.add((instance.purchased_vars[0] - pre_demand)*(pre_gamma*s_on + (1-pre_gamma)*s_off) <= instance.demand_disp_max)
        # for t in range(self.t_period-1):
        #     instance.displacement_bound.add((instance.purchased_vars[t+1] - instance.purchased_vars[t])*(gamma[t]*s_on + (1-gamma[t])*s_off) <= instance.demand_disp_max)

        #目的関数
        #総消費電力量、目標室温との誤差、消費電力変位の和を最小化
        #各項は事前の最適化計算の結果を用い、正規化
        def total_rule(instance):
            objective = weight*(((sum(instance.purchased_vars[t] for t in instance.IDX)**2) -best_cost**2) /(worst_cost**2 - best_cost**2))\
                + (1-weight)*(sum(occupancy[t]*(instance.bld_in_tmp_vars[t]-tmp_ref[t])**2 for t in instance.IDX)-best_tmp)/(worst_tmp - best_tmp)\
                + instance.pe_coeff*sum(instance.tmp_slack_vars[t] for t in instance.IDX)
            
            return objective
        instance.OBJ = Objective(rule=total_rule, sense=minimize)
        #ピーク項(抜いた)
        #+ instance.demand_disp_max/max_disp\

        #最適化
        self.opt.solve(instance,keepfiles=keepfiles, tee=stream_solver)

        tmp_prof = [value(instance.bld_in_tmp_vars[t]) for t in instance.IDX]
        demand_prof = [value(instance.purchased_vars[t]) for t in instance.IDX]
        switch_prof = [value(instance.switch[t]) for t in instance.IDX]
        return tmp_prof,demand_prof,switch_prof

#目的関数の正規化に使用するシミュレータ
class HVAC_Scheduler_sub:
    def __init__(self, bld_model, hvac_model, t_period, step_length=900, **kwargs):
    #bld_model: 建物モデル
    #hvac_model: 空調モデル
    #t_period: 計画期間のステップ数
    #step_length: 時間粒度[秒]
        #計画期間のステップ数
        self.t_period = t_period
        #時間粒度[秒]
        self.step_length = step_length
        #pyomoによる、最適化問題のモデル化
        self.mdl = AbstractModel()
        #建物モデル
        self.bld_params = bld_model.get_building_params()
        #空調モデル
        self.hvac_params = hvac_model.get_hvac_params()
        #使用するソルバー
        solver = "cplex"
        solver_io = "nl"
        self.opt = SolverFactory(solver)
        #self.opt.options['optimalitytarget'] = 3
        #self.opt = SolverFactory(solver, solver_io=solver_io)

        #最適化計算に時間制限を設ける
        self.opt.options['timelimit'] = 60*3

        if self.opt is None:
            print("")
            print("ERROR: Unable to create solver plugin for %s "
                  "using the %s interface" % (solver, solver_io))
            print("")
            exit(1)
        self.setup_abstmdl()

    def setup_abstmdl(self):
        mdl = self.mdl
        t_period = self.t_period

        # -- 変数の定義 -- #
        mdl.IDX = Set(initialize=range(t_period))
        #消費電力比
        mdl.demand_ratio = Var(mdl.IDX,bounds=(0,1), within=NonNegativeReals,initialize=0)
        #消費電力[kW]
        mdl.demand = Var(mdl.IDX, within=NonNegativeReals,initialize=0)
        #購入電力量[kWh]
        mdl.purchased_vars = Var(mdl.IDX, within=NonNegativeReals,initialize=0)
        #室温[℃]
        mdl.bld_in_tmp_vars = Var(range(t_period+1), within=Reals,initialize=0)
        #消費電力量の変位[kWh]
        mdl.demand_disp_max = Var(within=Reals,initialize=0)

        #空調のON/OFF
        #switchが 0の場合、off　1の場合、on
        mdl.switch = Var(mdl.IDX, within=Binary,initialize=0)


        mdl.tmp_slack_vars = Var(mdl.IDX, bounds=(0,5), within=Reals,initialize=0)
        mdl.pe_coeff = 1000000

    def optimize(self, ueq_list, get_column, tmp_ref, tmp_upper, tmp_lower,  bld_tmp_ini,tmp_in_past, out_tmp,out_tmp_past,demand_past,occupancy, weight):
    # def optimize(self, ueq_list, get_column, tmp_ref, tmp_upper, tmp_lower,  bld_tmp_ini,tmp_in_past, out_tmp,out_tmp_past,demand_past, occupancy, s_on,s_off,gamma,pre_gamma,pre_demand,max_disp,weight):
    # 2023-10-25変更
    #ueq_list: 室温予測式
    #get_column: 室温予測に使うデータのカラム
    #tmp_ref: 目標室温[℃]
    #tmp_upper: 室温の上限[℃]
    #tmp_lower: 室温の下限[℃]
    #bld_tmp_ini: 室内の初期温度[℃]
    #out_tmp: 外気温度[℃]
    #occupancy: 在室状況
    #weight: 目的関数の重み
        instance = self.mdl.create_instance()
        #計画期間における室温の初期値を設定
        instance.first_tmp = Constraint(expr = instance.bld_in_tmp_vars[0] == bld_tmp_ini)
        #室温の上限と下限を設定
        instance.tmp_bound = ConstraintList()
        for t in instance.IDX:
            instance.tmp_bound.add(instance.bld_in_tmp_vars[t] <= tmp_upper[t] + instance.tmp_slack_vars[t])
            instance.tmp_bound.add(instance.bld_in_tmp_vars[t] >= tmp_lower[t] - instance.tmp_slack_vars[t])

        '''
        #空調モデルの区分線形関数を設定
        for t in instance.IDX:
            #外気温が想定範囲から超えている場合
            #収まる値に変更
            ot = int(out_tmp[t])
            if ot < self.hvac_params['tmp_lower']:
                ot = self.hvac_params['tmp_lower']
            if ot > self.hvac_params['tmp_upper']:
                ot = self.hvac_params['tmp_upper']
            #区分線形関数を設定
            #動的に宣言しているため、exec関数を使用
            exec(f"instance.con_{t} = Piecewise(instance.demand_ratio[{t}],instance.power_ratio[{t}],pw_pts=self.hvac_params['xdata'][{ot-self.hvac_params['tmp_lower']}],pw_constr_type='EQ',f_rule=self.hvac_params['ydata'][{ot-self.hvac_params['tmp_lower']}],pw_repn='SOS2')")
        '''

        #消費電力に関する制約リスト
        instance.voltage_con = ConstraintList()
        for t in instance.IDX:
            #外気温が想定範囲から超えている場合
            #収まる値に変更
            ot = int(out_tmp[t])
            if ot < self.hvac_params['tmp_lower']:
                ot = self.hvac_params['tmp_lower']
            if ot > self.hvac_params['tmp_upper']:
                ot = self.hvac_params['tmp_upper']
            #購入電力量と消費電力比の関係
            instance.voltage_con.add(instance.purchased_vars[t] == instance.demand[t]*(self.step_length/3600))
            instance.voltage_con.add(instance.demand[t] == self.hvac_params['v_upper']*instance.demand_ratio[t])
            #消費電力の制約（追加）
            #instance.demand[t] 
            

            #switchが0の場合、power_ratioも0になり
            #switchが1の場合、power_ratioはth以上になる
            instance.voltage_con.add(instance.demand_ratio[t] >= th*instance.switch[t])
            instance.voltage_con.add(instance.demand_ratio[t] <= instance.switch[t])

        #室温変化の制約式
        instance.tmp_control = ConstraintList()
        for t in instance.IDX:
            mae = -1
            #print(get_column)
            for gc,eqa in zip(get_column,ueq_list):
                if t >= mae and t < gc:
                    print(f"{t=}")
                    print(f"{eqa=}")
                    exec(f"instance.tmp_control.add({eqa})")
                    break
                mae = gc

        # ピーク項の制約 2023-10-25削除
        # #消費電力変位の制約式
        # instance.displacement_bound = ConstraintList()
        # #変数gammaが1の場合、消費電力変位に係数を乗じる
        # #これによって、目的関数におけるウェイトを大きくする
        # instance.displacement_bound.add((instance.purchased_vars[0] - pre_demand)*(pre_gamma*s_on + (1-pre_gamma)*s_off) <= instance.demand_disp_max)
        # for t in range(self.t_period-1):
        #     instance.displacement_bound.add((instance.purchased_vars[t+1] - instance.purchased_vars[t])*(gamma[t]*s_on + (1-gamma[t])*s_off) <= instance.demand_disp_max)

        #目的関数
        def total_rule(instance):
            objective = weight*(sum(instance.purchased_vars[t] for t in instance.IDX)**2) \
                + (1-weight)*(sum(occupancy[t]*(instance.bld_in_tmp_vars[t]-tmp_ref[t])**2 for t in instance.IDX))\
                + instance.pe_coeff*sum(instance.tmp_slack_vars[t] for t in instance.IDX)
            return objective
        instance.OBJ = Objective(rule=total_rule, sense=minimize)
        #ピーク項(抜いた)
        #+ instance.demand_disp_max/max_disp\
        #最適化
        self.opt.solve(instance,keepfiles=keepfiles, tee=stream_solver)

        #以下の2つを、目的関数の正規化に利用
        #消費電力量の総和
        cost_prof =  sum(value(instance.purchased_vars[t]) for t in instance.IDX)
        #目標室温との誤差の二乗の総和
        tmp_prof = sum(occupancy[t]*(value(instance.bld_in_tmp_vars[t])-tmp_ref[t])**2 for t in instance.IDX)

        tmp_prof_value = [value(instance.bld_in_tmp_vars[t]) for t in instance.IDX]
        demand_prof_value = [value(instance.purchased_vars[t]) for t in instance.IDX]
        switch_prof = [value(instance.switch[t]) for t in instance.IDX]
        return cost_prof, tmp_prof, tmp_prof_value,demand_prof_value,switch_prof
