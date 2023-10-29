# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import pandas as pd


class Building(metaclass=ABCMeta):

    def __init__(self, nb_zone=1):
        # number of building zone
        self.nb_zone = nb_zone

    @abstractmethod
    def get_building_params(self):
        pass


class HVAC(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_hvac_params(self):
        pass


#建物モデルのパラメータを設定
class firstRC(Building):
    """ 1st-order RC network model
        Parmeters
        - 1 Resistance: R[K/W]
        - 1 Capacitance: C[J/K]
        - time constant: tau[s]
    """
    def __init__(self, nb_zone=1):
        super().__init__(nb_zone)
        bld_type = '1stRC'

        R = 0.00389971615331053
        C = 7879964.66865623
        tau = R*C

        self.params = {'type': bld_type, 'tau': tau, 'R': R,'C': C}

    def get_building_params(self):
        return self.params

class Hitachi_cool(HVAC):
    def __init__(self):
        hvac_type = "piecewise_cool"
        tmp_upper = 40
        tmp_lower = 0

        #定格消費電力[kW]
        v_upper = 21

        self.params = {'type': hvac_type,'v_upper':v_upper, 'tmp_upper': tmp_upper, 'tmp_lower': tmp_lower}

    def get_hvac_params(self):
        return self.params

class Hitachi_heat(HVAC):
    def __init__(self):
        hvac_type = "piecewise_cool"
        tmp_upper = 25
        tmp_lower = -15

        #定格消費電力[kW]
        v_upper = 22.5

        self.params = {'type': hvac_type, 'v_upper':v_upper, 'tmp_upper': tmp_upper, 'tmp_lower': tmp_lower}

    def get_hvac_params(self):
        return self.params

#冷房時の空調能力
class PiecewiseCOP_cool(HVAC):
    def __init__(self):
        hvac_type = "piecewise_cool"
        #空調負荷率と消費電力比の区分線形関数のデータ読み込み
        xdata = pd.read_csv('./input/piecewise/piecewise_cool_x.csv', squeeze=True, header=None).values.tolist()
        ydata = pd.read_csv('./input/piecewise/piecewise_cool_y.csv', squeeze=True, header=None).values.tolist()
        #外気温ごとに区分線形関数を用意しており、
        #その用意した外気温の上限と下限
        tmp_upper = 40
        tmp_lower = 0

        #定格出力[kW]
        p_upper = 33.5
        #定格消費電力[kW]
        v_upper = 9.9

        self.params = {'type': hvac_type, 'xdata':xdata, 'ydata':ydata, 'p_upper':p_upper, 'v_upper':v_upper, 'tmp_upper': tmp_upper, 'tmp_lower': tmp_lower}

    def get_hvac_params(self):
        return self.params

class PiecewiseCOP_heat(HVAC):
    def __init__(self):
        hvac_type = "piecewise_hot"
        #空調負荷率と消費電力比の区分線形関数のデータ読み込み
        xdata = pd.read_csv(f'./input/piecewise/piecewise_hot_x.csv', squeeze=True, header=None).values.tolist()
        ydata = pd.read_csv(f'./input/piecewise/piecewise_hot_y.csv', squeeze=True, header=None).values.tolist()
        #外気温ごとに区分線形関数を用意しており、
        #その用意した外気温の上限と下限
        tmp_upper = 25
        tmp_lower = -15

        #定格出力[kW]
        p_upper = 37.5
        #定格消費電力[kW]
        v_upper = 12.6

        self.params = {'type': hvac_type, 'xdata':xdata, 'ydata':ydata, 'p_upper':p_upper, 'v_upper':v_upper, 'tmp_upper': tmp_upper, 'tmp_lower': tmp_lower}

    def get_hvac_params(self):
        return self.params
