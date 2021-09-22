#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import unicode_literals

import cgi
import plotly.graph_objs as go
import math
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

gettext = cgi.FieldStorage()
user_choose = gettext.getfirst("user_choose", "empty")

# считываем данные, введенные пользователем


choose = 0

theta_ebeam = ['E8M54', 'E8M68']
cos_theta_eps = ['E14M15', 'E14M20', 'E14M55', 'E14M85', 'E9M85', 'E9M100', 'E9M107', 'E9M37', 'E9M52', 'E9M282']
cos_theta_ebeam = ['E14M117', 'E14M132', 'E14M140']

if user_choose == 'E8M54':
    choose = {'angle': 'theta', 'q2': '0.5', 'w': '1.23', 'theta': ['22.5', '37.5', '52.5'], 'ebeam': '1.515',
              'particle': 'Pin'}
elif user_choose == 'E8M68':
    choose = {'angle': 'theta', 'q2': '0.5', 'w': '1.51', 'theta': ['82.5', '97.5', '112.5'], 'ebeam': '1.515',
              'particle': 'Pin'}
elif user_choose == 'E14M15':
    choose = {'angle': 'cos(theta)', 'q2': '1.72', 'w': '1.43', 'cos_theta': ['0.1', '0.4', '0.7'], 'eps': '0.909',
              'particle': 'Pin'}
elif user_choose == 'E14M20':
    choose = {'angle': 'cos(theta)', 'q2': '1.72', 'w': '1.53', 'cos_theta': ['-0.3', '0.1', '0.3'], 'eps': '0.909',
              'particle': 'Pin'}
elif user_choose == 'E14M55':
    choose = {'angle': 'cos(theta)', 'q2': '2.05', 'w': '1.67', 'cos_theta': ['0.5', '0.7', '0.9'], 'eps': '0.8629',
              'particle': 'Pin'}
elif user_choose == 'E14M85':
    choose = {'angle': 'cos(theta)', 'q2': '2.91', 'w': '1.15', 'cos_theta': ['-0.3', '0.1', '0.3'], 'eps': '0.8787',
              'particle': 'Pin'}
elif user_choose == 'E14M117':
    choose = {'angle': 'cos(theta)', 'q2': '3.48', 'w': '1.23', 'cos_theta': ['-0.7', '-0.5', '-0.3'], 'ebeam': '5.75',
              'particle': 'Pin'}
elif user_choose == 'E14M132':
    choose = {'angle': 'cos(theta)', 'q2': '3.48', 'w': '1.53', 'cos_theta': ['-0.1', '0.1', '0.3'], 'ebeam': '5.75',
              'particle': 'Pin'}
elif user_choose == 'E14M140':
    choose = {'angle': 'cos(theta)', 'q2': '3.48', 'w': '1.69', 'cos_theta': ['-0.3', '-0.5', '-0.7'], 'ebeam': '5.75',
              'particle': 'Pin'}
elif user_choose == 'E9M85':
    choose = {'angle': 'cos(theta)', 'q2': '0.525', 'w': '1.22', 'cos_theta': ['0.5', '0.7', '0.9'], 'eps': '0.781',
              'particle': 'Pi0P'}
elif user_choose == 'E9M100':
    choose = {'angle': 'cos(theta)', 'q2': '0.525', 'w': '1.52', 'cos_theta': ['-0.3', '-0.1', '0.3'], 'eps': '0.517',
              'particle': 'Pi0P'}
elif user_choose == 'E9M107':
    choose = {'angle': 'cos(theta)', 'q2': '0.525', 'w': '1.66', 'cos_theta': ['-0.3', '-0.1', '0.1'], 'eps': '0.303',
              'particle': 'Pi0P'}
elif user_choose == 'E9M37':
    choose = {'angle': 'cos(theta)', 'q2': '1.45', 'w': '1.22', 'cos_theta': ['0.1', '0.3', '0.5'], 'eps': '0.689',
              'particle': 'Pi0P'}
elif user_choose == 'E9M52':
    choose = {'angle': 'cos(theta)', 'q2': '1.45', 'w': '1.52', 'cos_theta': ['0.3', '0.5', '0.7'], 'eps': '0.495',
              'particle': 'Pi0P'}
elif user_choose == 'E9M282':
    choose = {'angle': 'cos(theta)', 'q2': '1.15', 'w': '1.66', 'cos_theta': ['-0.3', '-0.4', '-0.5'], 'eps': '0.483',
              'particle': 'Pi0P'}




if user_choose in theta_ebeam:
    q2_value = choose['q2']
    w_value = choose['w']
    cos_value = str(math.cos(float(choose['theta'][0]) * 3.14 / 180))
    phi_value = 'empty'
    eps_value = 'empty'
    eBeamalue = choose['ebeam']
    particle = choose['particle']
    name_0 = 'theta=' + choose['theta'][0] + ' degrees'

    q2Value1 = choose['q2']
    wValue1 = choose['w']
    cosValue1 = str(math.cos(float(choose['theta'][1]) * 3.14 / 180))
    phiValue1 = 'empty'
    epsValue1 = 'empty'
    eBeamValue1 = choose['ebeam']
    particle1 = choose['particle']

    q2Value2 = choose['q2']
    wValue2 = choose['w']
    cosValue2 = str(math.cos(float(choose['theta'][2]) * 3.14 / 180))
    phiValue2 = 'empty'
    epsValue2 = 'empty'
    eBeamValue2 = choose['ebeam']
    particle2 = choose['particle']

    name_0 = 'theta=' + choose['theta'][0] + ' degrees'
    name_1 = 'theta=' + choose['theta'][1] + ' degrees'
    name_2 = 'theta=' + choose['theta'][2] + ' degrees'

    experimental_name = 'theta=' + choose['theta'][0] + ' degrees'
    experimental_name1 = 'theta=' + choose['theta'][1] + ' degrees'
    experimental_name2 = 'theta=' + choose['theta'][2] + ' degrees'

if user_choose in cos_theta_eps:
    q2Value = choose['q2']
    wValue = choose['w']
    cosValue = choose['cos_theta'][0]
    phiValue = 'empty'
    epsValue = choose['eps']
    eBeamValue = 'empty'
    particle = choose['particle']

    q2Value1 = choose['q2']
    wValue1 = choose['w']
    cosValue1 = choose['cos_theta'][1]
    phiValue1 = 'empty'
    epsValue1 = choose['eps']
    eBeamValue1 = 'empty'
    particle1 = choose['particle']

    q2Value2 = choose['q2']
    wValue2 = choose['w']
    cosValue2 = choose['cos_theta'][2]
    phiValue2 = 'empty'
    epsValue2 = choose['eps']
    eBeamValue2 = 'empty'
    particle2 = choose['particle']

    name_0 = 'cos(theta)=' + choose['cos_theta'][0] + ' degrees'
    name_1 = 'cos(theta)=' + choose['cos_theta'][1] + ' degrees'
    name_2 = 'cos(theta)=' + choose['cos_theta'][2] + ' degrees'

    experimental_name = 'cos(theta)=' + choose['cos_theta'][0] + ' degrees'
    experimental_name1 = 'cos(theta)=' + choose['cos_theta'][1] + ' degrees'
    experimental_name2 = 'cos(theta)=' + choose['cos_theta'][2] + ' degrees'

if user_choose in cos_theta_ebeam:
    q2Value = choose['q2']
    wValue = choose['w']
    cosValue = choose['cos_theta'][0]
    phiValue = 'empty'
    epsValue = 'empty'
    eBeamValue = choose['ebeam']
    particle = choose['particle']

    q2Value1 = choose['q2']
    wValue1 = choose['w']
    cosValue1 = choose['cos_theta'][1]
    phiValue1 = 'empty'
    epsValue1 = 'empty'
    eBeamValue1 = choose['ebeam']
    particle1 = choose['particle']

    q2Value2 = choose['q2']
    wValue2 = choose['w']
    cosValue2 = choose['cos_theta'][2]
    phiValue2 = 'empty'
    epsValue2 = 'empty'
    eBeamValue2 = choose['ebeam']
    particle2 = choose['particle']

    name_0 = 'cos(theta)=' + choose['cos_theta'][0] + ' degrees'
    name_1 = 'cos(theta)=' + choose['cos_theta'][1] + ' degrees'
    name_2 = 'cos(theta)=' + choose['cos_theta'][2] + ' degrees'

    experimental_name = 'cos(theta)=' + choose['cos_theta'][0] + ' degrees'
    experimental_name1 = 'cos(theta)=' + choose['cos_theta'][1] + ' degrees'
    experimental_name2 = 'cos(theta)=' + choose['cos_theta'][2] + ' degrees'


filename = 'data/' + user_choose + '/data.txt'

if user_choose in ['E9M107', 'E9M52']:
    if user_choose in theta_ebeam:
        exp_data = pd.read_csv(filename, sep='\t',
                               names=['theta', 'phi',
                                      'cross_section', 'd_cross_section_all'])

    if user_choose in cos_theta_eps:
        exp_data = pd.read_csv(filename, sep='\t',
                               names=['cos_theta', 'phi',
                                      'cross_section', 'd_cross_section_all'])

    if user_choose in cos_theta_ebeam:
        exp_data = pd.read_csv(filename, sep='\t',
                               names=['cos_theta', 'phi',
                                      'cross_section', 'd_cross_section_all'])

    if user_choose in theta_ebeam:
        exp_data = exp_data[exp_data['theta'] == float(choose['theta'][0])]

    if user_choose in cos_theta_eps:
        exp_data = exp_data[exp_data['cos_theta'] == float(choose['cos_theta'][0])]

    if user_choose in cos_theta_ebeam:
        exp_data = exp_data[exp_data['cos_theta'] == float(choose['cos_theta'][0])]

else:
    if user_choose in theta_ebeam:
        exp_data = pd.read_csv(filename, sep='\t',
                               names=['theta', 'phi',
                                      'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in cos_theta_eps:
        exp_data = pd.read_csv(filename, sep='\t',
                               names=['cos_theta', 'phi',
                                      'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in cos_theta_ebeam:
        exp_data = pd.read_csv(filename, sep='\t',
                               names=['cos_theta', 'phi',
                                      'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in theta_ebeam:
        exp_data = exp_data[exp_data['theta'] == float(choose['theta'][0])]

    if user_choose in cos_theta_eps:
        exp_data = exp_data[exp_data['cos_theta'] == float(choose['cos_theta'][0])]

    if user_choose in cos_theta_ebeam:
        exp_data = exp_data[exp_data['cos_theta'] == float(choose['cos_theta'][0])]

if user_choose in ['E9M107', 'E9M52']:
    if user_choose in theta_ebeam:
        exp_data1 = pd.read_csv(filename, sep='\t',
                                names=['theta', 'phi',
                                       'cross_section', 'd_cross_section_all'])

    if user_choose in cos_theta_eps:
        exp_data1 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'd_cross_section_all'])

    if user_choose in cos_theta_ebeam:
        exp_data1 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'd_cross_section_all'])

    if user_choose in theta_ebeam:
        exp_data1 = exp_data1[exp_data1['theta'] == float(choose['theta'][1])]

    if user_choose in cos_theta_eps:
        exp_data1 = exp_data1[exp_data1['cos_theta'] == float(choose['cos_theta'][1])]

    if user_choose in cos_theta_ebeam:
        exp_data1 = exp_data1[exp_data1['cos_theta'] == float(choose['cos_theta'][1])]
else:
    if user_choose in theta_ebeam:
        exp_data1 = pd.read_csv(filename, sep='\t',
                                names=['theta', 'phi',
                                       'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in cos_theta_eps:
        exp_data1 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in cos_theta_ebeam:
        exp_data1 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in theta_ebeam:
        exp_data1 = exp_data1[exp_data1['theta'] == float(choose['theta'][1])]

    if user_choose in cos_theta_eps:
        exp_data1 = exp_data1[exp_data1['cos_theta'] == float(choose['cos_theta'][1])]

    if user_choose in cos_theta_ebeam:
        exp_data1 = exp_data1[exp_data1['cos_theta'] == float(choose['cos_theta'][1])]

if user_choose in ['E9M107', 'E9M52']:
    if user_choose in theta_ebeam:
        exp_data2 = pd.read_csv(filename, sep='\t',
                                names=['theta', 'phi',
                                       'cross_section', 'd_cross_section_all'])

    if user_choose in cos_theta_eps:
        exp_data2 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'd_cross_section_all'])

    if user_choose in cos_theta_ebeam:
        exp_data2 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'd_cross_section_all'])

    if user_choose in theta_ebeam:
        exp_data2 = exp_data2[exp_data2['theta'] == float(choose['theta'][2])]

    if user_choose in cos_theta_eps:
        exp_data2 = exp_data2[exp_data2['cos_theta'] == float(choose['cos_theta'][2])]

    if user_choose in cos_theta_ebeam:
        exp_data2 = exp_data2[exp_data2['cos_theta'] == float(choose['cos_theta'][2])]

else:
    if user_choose in theta_ebeam:
        exp_data2 = pd.read_csv(filename, sep='\t',
                                names=['theta', 'phi',
                                       'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in cos_theta_eps:
        exp_data2 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in cos_theta_ebeam:
        exp_data2 = pd.read_csv(filename, sep='\t',
                                names=['cos_theta', 'phi',
                                       'cross_section', 'dcross_section', 'dcross_section_syst'])

    if user_choose in theta_ebeam:
        exp_data2 = exp_data2[exp_data2['theta'] == float(choose['theta'][2])]

    if user_choose in cos_theta_eps:
        exp_data2 = exp_data2[exp_data2['cos_theta'] == float(choose['cos_theta'][2])]

    if user_choose in cos_theta_ebeam:
        exp_data2 = exp_data2[exp_data2['cos_theta'] == float(choose['cos_theta'][2])]

    exp_data['d_cross_section_all'] = (exp_data['dcross_section'] ** 2 + exp_data['dcross_section_syst'] ** 2) ** 0.5

    exp_data1['d_cross_section_all'] = (exp_data1['dcross_section'] ** 2 + exp_data1['dcross_section_syst'] ** 2) ** 0.5

    exp_data2['d_cross_section_all'] = (exp_data2['dcross_section'] ** 2 + exp_data2['dcross_section_syst'] ** 2) ** 0.5

exp_data = exp_data.sort_values(by='phi')
exp_data1 = exp_data1.sort_values(by='phi')
exp_data2 = exp_data2.sort_values(by='phi')

pointsNum = 100
mp = 0.93827

df = pd.read_csv('final_table_cleaned.csv', header=None, sep='\t',
                 names=['Channel', 'MID',
                        'Wmin', 'Wmax', 'Q2min', 'Q2max', 'Cos(theta)',
                        'Sigma_T', 'dSigma_T', 'Sigma_L', 'dSigma_L',
                        'Sigma_TT', 'dSigma_TT', 'Sigma_LT', 'dSigma_LT',
                        'eps'])

a1new = (df['Wmin'] + df['Wmax']) / 2
a2new = (df['Q2min'] + df['Q2max']) / 2
df = df.assign(Waverage=a1new)
df = df.assign(Q2average=a2new)


class simpleMeasure(object):

    def __init__(self, wValueUser='empty', q2ValueUser='empty', cosValueUser='empty', ebeamValueUser='empty',
                 epsValueUser='empty', phiValueUser='empty', particleUser='Pin'):

        self.method = 0
        self.epsValue = epsValueUser.replace(',', '.')

        self.eps1 = 0
        self.eps2 = 0
        self.eps3 = 0
        self.eps4 = 0

        self.wValue = wValueUser
        self.q2Value = q2ValueUser
        self.cosValue = cosValueUser
        self.ebeamValue = ebeamValueUser
        self.phiValue = phiValueUser
        self.particle = particleUser

        self.checkwValue = True
        self.checkq2Value = True
        self.checkcosValue = True
        self.checkebeamValue = True
        self.checkphiValue = True
        self.checkepsValue = True

        self.wValue = self.wValue.replace(',', '.')
        self.q2Value = self.q2Value.replace(',', '.')
        self.cosValue = self.cosValue.replace(',', '.')
        self.ebeamValue = self.ebeamValue.replace(',', '.')
        self.phiValue = self.phiValue.replace(',', '.')

        try:
            self.wValue = float(self.wValue)
        except Exception:
            self.checkwValue = False
        try:
            self.q2Value = float(self.q2Value)
        except Exception:
            self.checkq2Value = False
        try:
            self.cosValue = float(self.cosValue)
        except Exception:
            self.checkcosValue = False
        try:
            self.phiValue = float(self.phiValue)
        except Exception:
            self.xasixValue = np.linspace(-1, 1, pointsNum)
            self.checkphiValue = False
        try:
            self.ebeamValue = float(self.ebeamValue)
        except Exception:
            self.checkebeamValue = False

        try:
            self.epsValue = float(self.epsValue)
        except Exception:
            self.checkepsValue = False

        self.xasixValue = 0
        self.xasixValue1 = 0
        self.xasixValue2 = 0
        self.xasixValue3 = 0
        self.xasixValue4 = 0

        self.sigma_TT = []
        self.sigma_LT = []
        self.sigma_T = []
        self.sigma_L = []
        self.dsigma_TT = []
        self.dsigma_LT = []
        self.dsigma_T = []
        self.dsigma_L = []
        self.resA = []
        self.resB = []
        self.resC = []
        self.dresA = []
        self.dresB = []
        self.dresC = []
        self.resCrossSect = []
        self.dresCrossSect = []

        self.sigma_TT1 = np.zeros(pointsNum)
        self.sigma_LT1 = np.zeros(pointsNum)
        self.sigma_T1 = np.zeros(pointsNum)
        self.sigma_L1 = np.zeros(pointsNum)
        self.dsigma_TT1 = np.zeros(pointsNum)
        self.dsigma_LT1 = np.zeros(pointsNum)
        self.dsigma_T1 = np.zeros(pointsNum)
        self.dsigma_L1 = np.zeros(pointsNum)
        self.resA1 = np.zeros(pointsNum)
        self.resB1 = np.zeros(pointsNum)
        self.resC1 = np.zeros(pointsNum)
        self.dresA1 = np.zeros(pointsNum)
        self.dresB1 = np.zeros(pointsNum)
        self.dresC1 = np.zeros(pointsNum)
        self.resCrossSect1 = np.zeros(pointsNum)
        self.dresCrossSect1 = np.zeros(pointsNum)

        self.sigma_TT2 = np.zeros(pointsNum)
        self.sigma_LT2 = np.zeros(pointsNum)
        self.sigma_T2 = np.zeros(pointsNum)
        self.sigma_L2 = np.zeros(pointsNum)
        self.dsigma_TT2 = np.zeros(pointsNum)
        self.dsigma_LT2 = np.zeros(pointsNum)
        self.dsigma_T2 = np.zeros(pointsNum)
        self.dsigma_L2 = np.zeros(pointsNum)
        self.resA2 = np.zeros(pointsNum)
        self.resB2 = np.zeros(pointsNum)
        self.resC2 = np.zeros(pointsNum)
        self.dresA2 = np.zeros(pointsNum)
        self.dresB2 = np.zeros(pointsNum)
        self.dresC2 = np.zeros(pointsNum)
        self.resCrossSect2 = np.zeros(pointsNum)
        self.dresCrossSect2 = np.zeros(pointsNum)

        self.sigma_TT3 = np.zeros(pointsNum)
        self.sigma_LT3 = np.zeros(pointsNum)
        self.sigma_T3 = np.zeros(pointsNum)
        self.sigma_L3 = np.zeros(pointsNum)
        self.dsigma_TT3 = np.zeros(pointsNum)
        self.dsigma_LT3 = np.zeros(pointsNum)
        self.dsigma_T3 = np.zeros(pointsNum)
        self.dsigma_L3 = np.zeros(pointsNum)
        self.resA3 = np.zeros(pointsNum)
        self.resB3 = np.zeros(pointsNum)
        self.resC3 = np.zeros(pointsNum)
        self.dresA3 = np.zeros(pointsNum)
        self.dresB3 = np.zeros(pointsNum)
        self.dresC3 = np.zeros(pointsNum)
        self.resCrossSect3 = np.zeros(pointsNum)
        self.dresCrossSect3 = np.zeros(pointsNum)

        self.sigma_TT4 = np.zeros(pointsNum)
        self.sigma_LT4 = np.zeros(pointsNum)
        self.sigma_T4 = np.zeros(pointsNum)
        self.sigma_L4 = np.zeros(pointsNum)
        self.dsigma_TT4 = np.zeros(pointsNum)
        self.dsigma_LT4 = np.zeros(pointsNum)
        self.dsigma_T4 = np.zeros(pointsNum)
        self.dsigma_L4 = np.zeros(pointsNum)
        self.resA4 = np.zeros(pointsNum)
        self.resB4 = np.zeros(pointsNum)
        self.resC4 = np.zeros(pointsNum)
        self.dresA4 = np.zeros(pointsNum)
        self.dresB4 = np.zeros(pointsNum)
        self.dresC4 = np.zeros(pointsNum)
        self.resCrossSect4 = np.zeros(pointsNum)
        self.dresCrossSect4 = np.zeros(pointsNum)

        self.XaxisStartIndex1 = 0
        self.XaxisFinishIndex1 = 0
        self.XaxisStartIndex2 = 0
        self.XaxisFinishIndex2 = 0
        self.XaxisStartIndex3 = 0
        self.XaxisFinishIndex3 = 0
        self.XaxisStartIndex4 = 0
        self.XaxisFinishIndex4 = 0

        self.xlabel = 0
        self.ylabel = 0

        if (self.particle == 'Pin'):
            self.PartNum = '1212'
            self.ParticleSecret = 'PIN'
            self.ParticleBeauty = 'gvp--->ПЂвЃєn'
            self.testdf = df[(df.Channel == 8) | (df.Channel == 14) | (df.Channel == 41) | (df.Channel == 141)]
            self.testdf1 = self.testdf[
                (self.testdf.Waverage >= 1.1) & (self.testdf.Waverage <= 1.4) & (self.testdf.Q2average >= 0.2) & (
                            self.testdf.Q2average <= 0.7)]
            self.testdf2 = self.testdf[
                (self.testdf.Waverage >= 1.4) & (self.testdf.Waverage <= 1.6) & (self.testdf.Q2average >= 0.2) & (
                            self.testdf.Q2average <= 0.7)]
            self.testdf3 = self.testdf[
                (self.testdf.Waverage >= 1.1) & (self.testdf.Waverage <= 1.6) & (self.testdf.Q2average >= 1.5) & (
                            self.testdf.Q2average <= 5)]
            self.testdf4 = self.testdf[
                (self.testdf.Waverage >= 1.6) & (self.testdf.Waverage <= 1.9) & (self.testdf.Q2average >= 1.5) & (
                            self.testdf.Q2average <= 5)]
        if (self.particle == 'Pi0P'):
            self.PartNum = '1213'
            self.ParticleSecret = 'PI0P'
            self.ParticleBeauty = 'gvp--->ПЂвЃ°p'
            self.testdf = df[(df.Channel == 9) | (df.Channel == 37) | (df.Channel == 170)]
            self.testdf1 = self.testdf[
                (self.testdf.Waverage >= 1.1) & (self.testdf.Waverage <= 1.7875) & (self.testdf.Q2average >= 0.4) & (
                            self.testdf.Q2average <= 1.8)]
            self.testdf2 = self.testdf[
                (self.testdf.Waverage >= 1.11) & (self.testdf.Waverage <= 1.39) & (self.testdf.Q2average >= 3) & (
                            self.testdf.Q2average <= 6)]
            self.testdf3 = self.testdf[(self.testdf.Waverage == 0)]
            self.testdf4 = self.testdf[(self.testdf.Waverage == 0)]

        self.a1 = np.array(self.testdf1['Waverage'])
        self.b1 = np.array(self.testdf1['Q2average'])
        self.c1 = np.array(self.testdf1['Cos(theta)'])
        self.d1 = np.array(self.testdf1['Sigma_TT'])
        self.e1 = np.array(self.testdf1['Sigma_LT'])
        self.f1 = np.array(self.testdf1['Sigma_T'])
        self.g1 = np.array(self.testdf1['Sigma_L'])
        self.dd1 = np.array(self.testdf1['dSigma_TT'])
        self.ee1 = np.array(self.testdf1['dSigma_LT'])
        self.ff1 = np.array(self.testdf1['dSigma_T'])
        self.gg1 = np.array(self.testdf1['dSigma_L'])

        self.a2 = np.array(self.testdf2['Waverage'])
        self.b2 = np.array(self.testdf2['Q2average'])
        self.c2 = np.array(self.testdf2['Cos(theta)'])
        self.d2 = np.array(self.testdf2['Sigma_TT'])
        self.e2 = np.array(self.testdf2['Sigma_LT'])
        self.f2 = np.array(self.testdf2['Sigma_T'])
        self.g2 = np.array(self.testdf2['Sigma_L'])
        self.dd2 = np.array(self.testdf2['dSigma_TT'])
        self.ee2 = np.array(self.testdf2['dSigma_LT'])
        self.ff2 = np.array(self.testdf2['dSigma_T'])
        self.gg2 = np.array(self.testdf2['dSigma_L'])

        if (self.particle != 'Pi0P'):
            self.a3 = np.array(self.testdf3['Waverage'])
            self.b3 = np.array(self.testdf3['Q2average'])
            self.c3 = np.array(self.testdf3['Cos(theta)'])
            self.d3 = np.array(self.testdf3['Sigma_TT'])
            self.e3 = np.array(self.testdf3['Sigma_LT'])
            self.f3 = np.array(self.testdf3['Sigma_T'])
            self.g3 = np.array(self.testdf3['Sigma_L'])
            self.dd3 = np.array(self.testdf3['dSigma_TT'])
            self.ee3 = np.array(self.testdf3['dSigma_LT'])
            self.ff3 = np.array(self.testdf3['dSigma_T'])
            self.gg3 = np.array(self.testdf3['dSigma_L'])

            self.a4 = np.array(self.testdf4['Waverage'])
            self.b4 = np.array(self.testdf4['Q2average'])
            self.c4 = np.array(self.testdf4['Cos(theta)'])
            self.d4 = np.array(self.testdf4['Sigma_TT'])
            self.e4 = np.array(self.testdf4['Sigma_LT'])
            self.f4 = np.array(self.testdf4['Sigma_T'])
            self.g4 = np.array(self.testdf4['Sigma_L'])
            self.dd4 = np.array(self.testdf4['dSigma_TT'])
            self.ee4 = np.array(self.testdf4['dSigma_LT'])
            self.ff4 = np.array(self.testdf4['dSigma_T'])
            self.gg4 = np.array(self.testdf4['dSigma_L'])

    def interpolateGraphAndError(self, wValue, q2Value, cosValue):
        self.sigma_TT1 = griddata((self.a1, self.b1, self.c1), self.d1, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)
        self.sigma_LT1 = griddata((self.a1, self.b1, self.c1), self.e1, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)
        self.sigma_T1 = griddata((self.a1, self.b1, self.c1), self.f1, (wValue, q2Value, cosValue), method='linear',
                                 rescale=True)
        self.sigma_L1 = griddata((self.a1, self.b1, self.c1), self.g1, (wValue, q2Value, cosValue), method='linear',
                                 rescale=True)
        self.dsigma_TT1 = griddata((self.a1, self.b1, self.c1), self.dd1, (wValue, q2Value, cosValue), method='linear',
                                   rescale=True)
        self.dsigma_LT1 = griddata((self.a1, self.b1, self.c1), self.ee1, (wValue, q2Value, cosValue), method='linear',
                                   rescale=True)
        self.dsigma_T1 = griddata((self.a1, self.b1, self.c1), self.ff1, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)
        self.dsigma_L1 = griddata((self.a1, self.b1, self.c1), self.gg1, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)

        self.sigma_TT2 = griddata((self.a2, self.b2, self.c2), self.d2, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)
        self.sigma_LT2 = griddata((self.a2, self.b2, self.c2), self.e2, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)
        self.sigma_T2 = griddata((self.a2, self.b2, self.c2), self.f2, (wValue, q2Value, cosValue), method='linear',
                                 rescale=True)
        self.sigma_L2 = griddata((self.a2, self.b2, self.c2), self.g2, (wValue, q2Value, cosValue), method='linear',
                                 rescale=True)
        self.dsigma_TT2 = griddata((self.a2, self.b2, self.c2), self.dd2, (wValue, q2Value, cosValue), method='linear',
                                   rescale=True)
        self.dsigma_LT2 = griddata((self.a2, self.b2, self.c2), self.ee2, (wValue, q2Value, cosValue), method='linear',
                                   rescale=True)
        self.dsigma_T2 = griddata((self.a2, self.b2, self.c2), self.ff2, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)
        self.dsigma_L2 = griddata((self.a2, self.b2, self.c2), self.gg2, (wValue, q2Value, cosValue), method='linear',
                                  rescale=True)

        if (self.particle != 'Pi0P'):
            self.sigma_TT3 = griddata((self.a3, self.b3, self.c3), self.d3, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)
            self.sigma_LT3 = griddata((self.a3, self.b3, self.c3), self.e3, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)
            self.sigma_T3 = griddata((self.a3, self.b3, self.c3), self.f3, (wValue, q2Value, cosValue), method='linear',
                                     rescale=True)
            self.sigma_L3 = griddata((self.a3, self.b3, self.c3), self.g3, (wValue, q2Value, cosValue), method='linear',
                                     rescale=True)
            self.dsigma_TT3 = griddata((self.a3, self.b3, self.c3), self.dd3, (wValue, q2Value, cosValue),
                                       method='linear', rescale=True)
            self.dsigma_LT3 = griddata((self.a3, self.b3, self.c3), self.ee3, (wValue, q2Value, cosValue),
                                       method='linear', rescale=True)
            self.dsigma_T3 = griddata((self.a3, self.b3, self.c3), self.ff3, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)
            self.dsigma_L3 = griddata((self.a3, self.b3, self.c3), self.gg3, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)

            self.sigma_TT4 = griddata((self.a4, self.b4, self.c4), self.d4, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)
            self.sigma_LT4 = griddata((self.a4, self.b4, self.c4), self.e4, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)
            self.sigma_T4 = griddata((self.a4, self.b4, self.c4), self.f4, (wValue, q2Value, cosValue), method='linear',
                                     rescale=True)
            self.sigma_L4 = griddata((self.a4, self.b4, self.c4), self.g4, (wValue, q2Value, cosValue), method='linear',
                                     rescale=True)
            self.dsigma_TT4 = griddata((self.a4, self.b4, self.c4), self.dd4, (wValue, q2Value, cosValue),
                                       method='linear', rescale=True)
            self.dsigma_LT4 = griddata((self.a4, self.b4, self.c4), self.ee4, (wValue, q2Value, cosValue),
                                       method='linear', rescale=True)
            self.dsigma_T4 = griddata((self.a4, self.b4, self.c4), self.ff4, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)
            self.dsigma_L4 = griddata((self.a4, self.b4, self.c4), self.gg4, (wValue, q2Value, cosValue),
                                      method='linear', rescale=True)

    def calculateEps(self, wValue, q2Value, ebeamValue):

        mp = 0.93827

        if (self.checkebeamValue and (not self.checkepsValue)):
            nu1 = (wValue * wValue + q2Value - mp * mp) / (2 * mp)
            self.eps1 = 1 / (1 + 2 * (nu1 * nu1 + q2Value) / (4 * (ebeamValue - nu1) * ebeamValue - q2Value))

            nu2 = (wValue * wValue + q2Value - mp * mp) / (2 * mp)
            self.eps2 = 1 / (1 + 2 * (nu2 * nu2 + q2Value) / (4 * (ebeamValue - nu2) * ebeamValue - q2Value))

            if (self.particle != 'Pi0P'):
                nu3 = (wValue * wValue + q2Value - mp * mp) / (2 * mp)
                self.eps3 = 1 / (1 + 2 * (nu3 * nu3 + q2Value) / (4 * (ebeamValue - nu3) * ebeamValue - q2Value))

                nu4 = (wValue * wValue + q2Value - mp * mp) / (2 * mp)
                self.eps4 = 1 / (1 + 2 * (nu4 * nu4 + q2Value) / (4 * (ebeamValue - nu4) * ebeamValue - q2Value))

        if (self.checkepsValue and (not self.checkebeamValue)):
            self.eps1 = np.empty(pointsNum)
            self.eps1.fill(self.epsValue)

            self.eps2 = np.empty(pointsNum)
            self.eps2.fill(self.epsValue)

            if (self.particle != 'Pi0P'):
                self.eps3 = np.empty(pointsNum)
                self.eps3.fill(self.epsValue)

                self.eps4 = np.empty(pointsNum)
                self.eps4.fill(self.epsValue)

    def cutData(self):

        nans1 = np.isnan(self.sigma_TT1)
        self.sigma_TT1 = self.sigma_TT1[nans1 == False]
        self.sigma_LT1 = self.sigma_LT1[nans1 == False]
        self.sigma_T1 = self.sigma_T1[nans1 == False]
        self.sigma_L1 = self.sigma_L1[nans1 == False]
        self.dsigma_TT1 = self.dsigma_TT1[nans1 == False]
        self.dsigma_LT1 = self.dsigma_LT1[nans1 == False]
        self.dsigma_T1 = self.dsigma_T1[nans1 == False]
        self.dsigma_L1 = self.dsigma_L1[nans1 == False]
        self.xasixValue1 = self.xasixValue1[nans1 == False]
        if (type(self.eps1) == np.ndarray):
            self.eps1 = self.eps1[nans1 == False]

        nans2 = np.isnan(self.sigma_TT2)
        self.sigma_TT2 = self.sigma_TT2[nans2 == False]
        self.sigma_LT2 = self.sigma_LT2[nans2 == False]
        self.sigma_T2 = self.sigma_T2[nans2 == False]
        self.sigma_L2 = self.sigma_L2[nans2 == False]
        self.dsigma_TT2 = self.dsigma_TT2[nans2 == False]
        self.dsigma_LT2 = self.dsigma_LT2[nans2 == False]
        self.dsigma_T2 = self.dsigma_T2[nans2 == False]
        self.dsigma_L2 = self.dsigma_L2[nans2 == False]
        self.xasixValue2 = self.xasixValue2[nans2 == False]
        if (type(self.eps2) == np.ndarray):
            self.eps2 = self.eps2[nans2 == False]

        if (self.particle != 'Pi0P'):
            nans3 = np.isnan(self.sigma_TT3)
            self.sigma_TT3 = self.sigma_TT3[nans3 == False]
            self.sigma_LT3 = self.sigma_LT3[nans3 == False]
            self.sigma_T3 = self.sigma_T3[nans3 == False]
            self.sigma_L3 = self.sigma_L3[nans3 == False]
            self.dsigma_TT3 = self.dsigma_TT3[nans3 == False]
            self.dsigma_LT3 = self.dsigma_LT3[nans3 == False]
            self.dsigma_T3 = self.dsigma_T3[nans3 == False]
            self.dsigma_L3 = self.dsigma_L3[nans3 == False]
            self.xasixValue3 = self.xasixValue3[nans3 == False]
            if (type(self.eps3) == np.ndarray):
                self.eps3 = self.eps3[nans3 == False]

            nans4 = np.isnan(self.sigma_TT4)
            self.sigma_TT4 = self.sigma_TT4[nans4 == False]
            self.sigma_LT4 = self.sigma_LT4[nans4 == False]
            self.sigma_T4 = self.sigma_T4[nans4 == False]
            self.sigma_L4 = self.sigma_L4[nans4 == False]
            self.dsigma_TT4 = self.dsigma_TT4[nans4 == False]
            self.dsigma_LT4 = self.dsigma_LT4[nans4 == False]
            self.dsigma_T4 = self.dsigma_T4[nans4 == False]
            self.dsigma_L4 = self.dsigma_L4[nans4 == False]
            self.xasixValue4 = self.xasixValue4[nans4 == False]
            if (type(self.eps4) == np.ndarray):
                self.eps4 = self.eps4[nans4 == False]

    def calculate_resABC_dAdBdC(self):
        self.resA1 = self.sigma_T1 + self.eps1 * self.sigma_L1
        self.dresA1 = ((self.dsigma_T1 ** 2) + ((self.eps1 * self.dsigma_L1) ** 2)) ** 0.5
        self.resB1 = self.eps1 * self.sigma_TT1
        self.dresB1 = self.eps1 * self.dsigma_TT1
        self.resC1 = ((2 * self.eps1 * (self.eps1 + 1)) ** 0.5) * self.sigma_LT1
        self.dresC1 = ((2 * self.eps1 * (self.eps1 + 1)) ** 0.5) * self.dsigma_LT1

        self.resA2 = self.sigma_T2 + self.eps2 * self.sigma_L2
        self.dresA2 = ((self.dsigma_T2 ** 2) + ((self.eps2 * self.dsigma_L2) ** 2)) ** 0.5
        self.resB2 = self.eps2 * self.sigma_TT2
        self.dresB2 = self.eps2 * self.dsigma_TT2
        self.resC2 = ((2 * self.eps2 * (self.eps2 + 1)) ** 0.5) * self.sigma_LT2
        self.dresC2 = ((2 * self.eps2 * (self.eps2 + 1)) ** 0.5) * self.dsigma_LT2

        if (self.particle != 'Pi0P'):
            self.resA3 = self.sigma_T3 + self.eps3 * self.sigma_L3
            self.dresA3 = ((self.dsigma_T3 ** 2) + ((self.eps3 * self.dsigma_L3) ** 2)) ** 0.5
            self.resB3 = self.eps3 * self.sigma_TT3
            self.dresB3 = self.eps3 * self.dsigma_TT3
            self.resC3 = ((2 * self.eps3 * (self.eps3 + 1)) ** 0.5) * self.sigma_LT3
            self.dresC3 = ((2 * self.eps3 * (self.eps3 + 1)) ** 0.5) * self.dsigma_LT3

            self.resA4 = self.sigma_T4 + self.eps4 * self.sigma_L4
            self.dresA4 = ((self.dsigma_T4 ** 2) + ((self.eps4 * self.dsigma_L4) ** 2)) ** 0.5
            self.resB4 = self.eps4 * self.sigma_TT4
            self.dresB4 = self.eps4 * self.dsigma_TT4
            self.resC4 = ((2 * self.eps4 * (self.eps4 + 1)) ** 0.5) * self.sigma_LT4
            self.dresC4 = ((2 * self.eps4 * (self.eps4 + 1)) ** 0.5) * self.dsigma_LT4

    def resCrossSect_dresCrossSect(self, phi):
        phi = phi * (np.pi / 180)

        if not (np.isnan(self.resA1).all()):
            self.resCrossSect1 = self.resA1 + self.resB1 * np.cos(2 * phi) + self.resC1 * np.cos(phi)
            self.dresCrossSect1 = (self.dresA1 ** 2 + (self.dresB1 * np.cos(2 * phi)) ** 2 + (
                        self.dresC1 * np.cos(phi)) ** 2) ** 0.5
        if (np.isnan(self.resA1).all()):
            self.resCrossSect1 = []
            self.dresCrossSect1 = []

        if not (np.isnan(self.resA2).all()):
            self.resCrossSect2 = self.resA2 + self.resB2 * np.cos(2 * phi) + self.resC2 * np.cos(phi)
            self.dresCrossSect2 = (self.dresA2 ** 2 + (self.dresB2 * np.cos(2 * phi)) ** 2 + (
                        self.dresC2 * np.cos(phi)) ** 2) ** 0.5
        if (np.isnan(self.resA2).all()):
            self.resCrossSect2 = []
            self.dresCrossSect2 = []

        if (self.particle != 'Pi0P'):

            if not (np.isnan(self.resA3).all()):
                self.resCrossSect3 = self.resA3 + self.resB3 * np.cos(2 * phi) + self.resC3 * np.cos(phi)
                self.dresCrossSect3 = (self.dresA3 ** 2 + (self.dresB3 * np.cos(2 * phi)) ** 2 + (
                            self.dresC3 * np.cos(phi)) ** 2) ** 0.5
            if (np.isnan(self.resA3).all()):
                self.resCrossSect3 = []
                self.dresCrossSect3 = []

            if not (np.isnan(self.resA4).all()):
                self.resCrossSect4 = self.resA4 + self.resB4 * np.cos(2 * phi) + self.resC4 * np.cos(phi)
                self.dresCrossSect4 = (self.dresA4 ** 2 + (self.dresB4 * np.cos(2 * phi)) ** 2 + (
                            self.dresC4 * np.cos(phi)) ** 2) ** 0.5
            if (np.isnan(self.resA4).all()):
                self.resCrossSect4 = []
                self.dresCrossSect4 = []

    def getEnteredDataOneTable(self):
        print("""<table border="1" width="1000px">
                <tr>
                    <th>$$Reaction$$</th>
                    <th>$$W(GeV)$$</th>
                    <th>$$Q^2(GeV^2)$$</th>
                    <th>$$cos(\\theta)$$</th>
                    <th>$$E(GeV)$$</th>
                    <th>$$\\varphi(degree)$$</th>
                </tr>""")
        print("""
            <tr>
                <td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>
            <tr>
                </table> <br> <br>""".format(self.ParticleBeauty, self.wValue, self.q2Value, self.cosValue,
                                             self.ebeamValue, self.phiValue))

    def getEnteredDataTwoTable(self, secondObject):
        print("""<table border="1" width="1000px">
                <tr>
                    <th>$$Reaction$$</th>
                    <th>$$W(GeV)$$</th>
                    <th>$$Q^2(GeV^2)$$</th>
                    <th>$$cos(\\theta)$$</th>
                    <th>$$E(GeV)$$</th>
                    <th>$$\\varphi(degree)$$</th>
                </tr>""")
        print("""
            <tr>
                <td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>
            <tr>
            <tr>
                <td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>
            <tr>
                </table> <br> <br>""".format(self.ParticleBeauty, self.wValue, self.q2Value, self.cosValue,
                                             self.ebeamValue, self.phiValue, secondObject.ParticleBeauty,
                                             secondObject.wValue, secondObject.q2Value, secondObject.cosValue,
                                             secondObject.ebeamValue, secondObject.phiValue))

    def getTT_TL_T_L_tableOne(self):
        print("""<table border="1" width="1000px">
                <tr>
                    <th>$$\dfrac{d\sigma_{TT}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{LT}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{T}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{L}}{d\Omega}$$</th>
                </tr>""")
        print("""
            <tr>
                <td>{} &#8723; {} </td><td>{} &#8723; {}</td><td>{} &#8723; {}</td><td>{} &#8723; {}</td>
            <tr>
                </table>""".format(round(float(self.sigma_TT), 5), round(float(self.dsigma_TT), 5),
                                   round(float(self.sigma_LT), 5), round(float(self.dsigma_LT), 5),
                                   round(float(self.sigma_T), 5), round(float(self.dsigma_T), 5),
                                   round(float(self.sigma_L), 5), round(float(self.dsigma_L), 5)))

    def getTT_TL_T_L_tableTwo(self, objc):
        print("""<table border="1" width="1000px">
                <tr>
                    <th>$$\dfrac{d\sigma_{TT}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{LT}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{T}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{L}}{d\Omega}$$</th>
                </tr>""")
        print("""
            <tr>
                <td>{} &#8723; {} </td><td>{} &#8723; {}</td><td>{} &#8723; {}</td><td>{} &#8723; {}</td>
            <tr>
            <tr>
                <td>{} &#8723; {} </td><td>{} &#8723; {}</td><td>{} &#8723; {}</td><td>{} &#8723; {}</td>
            <tr>
                </table>""".format(round(float(self.sigma_TT), 5), round(float(self.dsigma_TT), 5),
                                   round(float(self.sigma_LT), 5), round(float(self.dsigma_LT), 5),
                                   round(float(self.sigma_T), 5), round(float(self.dsigma_T), 5),
                                   round(float(self.sigma_L), 5), round(float(self.dsigma_L), 5),
                                   round(float(objc.sigma_TT), 5), round(float(objc.dsigma_TT), 5),
                                   round(float(objc.sigma_LT), 5), round(float(objc.dsigma_LT), 5),
                                   round(float(objc.sigma_T), 5), round(float(objc.dsigma_T), 5),
                                   round(float(objc.sigma_L), 5), round(float(objc.dsigma_L), 5)))

    def getCross_resA_TT_LT_tableOne(self):
        print("""<table border="1" width="1000px">
                <tr>
                    <th>$$\dfrac{d\sigma}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{U}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{TT}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{LT}}{d\Omega}$$</th>
                </tr>""")
        print("""
            <tr>
                <td>{} &#8723; {} </td><td>{} &#8723; {}</td><td>{} &#8723; {}</td><td>{} &#8723; {}</td>
            <tr>
                </table>""".format(round(float(self.resCrossSect), 5), round(float(self.dresCrossSect), 5),
                                   round(float(self.resA), 5), round(float(self.dresA), 5),
                                   round(float(self.sigma_TT), 5), round(float(self.dsigma_TT), 5),
                                   round(float(self.sigma_LT), 5), round(float(self.dsigma_LT), 5)))

    def getCross_resA_TT_LT_tableTwo(self, objc):
        print("""<table border="1" width="1000px">
                <tr>
                    <th>$$\dfrac{d\sigma}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{U}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{TT}}{d\Omega}$$</th>
                    <th>$$\dfrac{d\sigma_{LT}}{d\Omega}$$</th>
                </tr>""")
        print("""
            <tr>
                <td>{} &#8723; {} </td><td>{} &#8723; {}</td><td>{} &#8723; {}</td><td>{} &#8723; {}</td>
            <tr>
            <tr>
                <td>{} &#8723; {} </td><td>{} &#8723; {}</td><td>{} &#8723; {}</td><td>{} &#8723; {}</td>
            <tr>
                </table>""".format(round(float(self.resCrossSect), 5), round(float(self.dresCrossSect), 5),
                                   round(float(self.resA), 5), round(float(self.dresA), 5),
                                   round(float(self.sigma_TT), 5), round(float(self.dsigma_TT), 5),
                                   round(float(self.sigma_LT), 5), round(float(self.dsigma_LT), 5),
                                   round(float(objc.resCrossSect), 5), round(float(objc.dresCrossSect), 5),
                                   round(float(objc.resA), 5), round(float(objc.dresA), 5),
                                   round(float(objc.sigma_TT), 5), round(float(objc.dsigma_TT), 5),
                                   round(float(objc.sigma_LT), 5), round(float(objc.dsigma_LT), 5)))

        # Ввели WQCOS - method 1
        # ничего не строим, просто считаем значения (TT LT T L) в точке
        # Ввели WQCOS и ebeam  - method 11
        # тогда строим зависимости сечения и структурных фунций от угла phi
        # Ввели WQCOS и phi  - method 12
        # мдааааа, ничего не строим, рисуем табличку с структурными функциями в точке
        # Ввели WQCOS и ebeam и phi  - method 13
        # мдааааа, ничего не строим, рисуем табличку с сечением и структурными функциями в точке

        # Ввели WQ/QCOS/WCOS - method 2/3/4
        # строим зависимость только структурных функций от переменной Default
        # Ввели WQ/QCOS/WCOS и ebeam  - method 21/31/41
        # строим зависимость сечений и структурных функций (A, TT, TL) от переменной Default
        # Ввели WQ/QCOS/WCOS и phi  - method 22/32/44
        # строим зависимость структурных функций (TT, TL) от переменной Default
        # Ввели WQ/QCOS/WCOS и ebeam и phi  - method 23/33/43
        # строим зависимость сечения и структурных функций (A, TT, TL) от переменной Default

    # calculateEps(self,wValue,q2Value,ebeamValue)

    def findAndUseMethod(self):
        if (self.checkwValue and self.checkq2Value and self.checkcosValue):
            self.xlabel = '\u03c6(degree)'
            self.xasixValue = np.linspace(0, 360, pointsNum)
            self.xasixValue1 = self.xasixValue
            self.xasixValue2 = self.xasixValue
            self.xasixValue3 = self.xasixValue
            self.xasixValue4 = self.xasixValue
            if (not (self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 1
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.cosValue)
            if ((self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 11
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.cosValue)
                self.calculateEps(self.wValue, self.q2Value, self.ebeamValue)
                self.calculate_resABC_dAdBdC()
                self.resCrossSect_dresCrossSect(self.xasixValue)

            if (self.checkphiValue and not (self.checkebeamValue or self.checkepsValue)):
                self.method = 12
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.cosValue)

            if (self.checkphiValue and (self.checkebeamValue or self.checkepsValue)):
                self.method = 13
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.cosValue)
                self.calculateEps(self.wValue, self.q2Value, self.ebeamValue)
                self.calculate_resABC_dAdBdC()
                self.resCrossSect_dresCrossSect(self.xasixValue)

        if (self.checkwValue and self.checkq2Value and (not self.checkcosValue)):
            self.xlabel = 'cos(\u03b8)'
            self.xasixValue = np.linspace(-1, 1, pointsNum)
            self.xasixValue1 = self.xasixValue
            self.xasixValue2 = self.xasixValue
            self.xasixValue3 = self.xasixValue
            self.xasixValue4 = self.xasixValue
            if (not (self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 2
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.xasixValue)
                self.cutData()

            if ((self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 21
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.xasixValue)
                self.calculateEps(self.wValue, self.q2Value, self.ebeamValue)
                self.cutData()
                self.calculate_resABC_dAdBdC()

            if (self.checkphiValue and not (self.checkebeamValue or self.checkepsValue)):
                self.method = 22
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.xasixValue)
                self.cutData()

            if (self.checkphiValue and (self.checkebeamValue or self.checkepsValue)):
                self.method = 23
                self.interpolateGraphAndError(self.wValue, self.q2Value, self.xasixValue)
                self.calculateEps(self.wValue, self.q2Value, self.ebeamValue)
                self.cutData()
                self.calculate_resABC_dAdBdC()
                self.resCrossSect_dresCrossSect(self.phiValue)

        if ((not self.checkwValue) and self.checkq2Value and self.checkcosValue):
            self.xlabel = 'W(GeV)'
            self.xasixValue = np.linspace(1.1, 4.5, pointsNum)
            self.xasixValue1 = self.xasixValue
            self.xasixValue2 = self.xasixValue
            self.xasixValue3 = self.xasixValue
            self.xasixValue4 = self.xasixValue
            if (not (self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 3
                self.interpolateGraphAndError(self.xasixValue, self.q2Value, self.cosValue)
                self.cutData()

            if ((self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 31
                self.interpolateGraphAndError(self.xasixValue, self.q2Value, self.cosValue)
                self.calculateEps(self.xasixValue, self.q2Value, self.ebeamValue)
                self.cutData()
                self.calculate_resABC_dAdBdC()

            if (self.checkphiValue and not (self.checkebeamValue or self.checkepsValue)):
                self.method = 32
                self.interpolateGraphAndError(self.xasixValue, self.q2Value, self.cosValue)
                self.cutData()

            if (self.checkphiValue and (self.checkebeamValue or self.checkepsValue)):
                self.method = 33
                self.interpolateGraphAndError(self.xasixValue, self.q2Value, self.cosValue)
                self.calculateEps(self.xasixValue, self.q2Value, self.ebeamValue)
                self.cutData()
                self.calculate_resABC_dAdBdC()
                self.resCrossSect_dresCrossSect(self.phiValue)

        if (self.checkwValue and (not self.checkq2Value) and self.checkcosValue):
            self.xlabel = 'Q<sup>2</sup> (GeV<sup>2</sup>)'
            self.xasixValue = np.linspace(0.1, 4.5, pointsNum)
            self.xasixValue1 = self.xasixValue
            self.xasixValue2 = self.xasixValue
            self.xasixValue3 = self.xasixValue
            self.xasixValue4 = self.xasixValue
            if (not (self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 4
                self.interpolateGraphAndError(self.wValue, self.xasixValue, self.cosValue)
                self.cutData()

            if ((self.checkebeamValue or self.checkepsValue) and (not self.checkphiValue)):
                self.method = 41
                self.interpolateGraphAndError(self.wValue, self.xasixValue, self.cosValue)
                self.calculateEps(self.wValue, self.xasixValue, self.ebeamValue)
                self.cutData()
                self.calculate_resABC_dAdBdC()

            if (self.checkphiValue and not (self.checkebeamValue or self.checkepsValue)):
                self.method = 42
                self.interpolateGraphAndError(self.wValue, self.xasixValue, self.cosValue)
                self.cutData()

            if (self.checkphiValue and (self.checkebeamValue or self.checkepsValue)):
                self.method = 43
                self.interpolateGraphAndError(self.wValue, self.xasixValue, self.cosValue)
                self.calculateEps(self.wValue, self.xasixValue, self.ebeamValue)
                self.cutData()
                self.calculate_resABC_dAdBdC()
                self.resCrossSect_dresCrossSect(self.phiValue)

    def unionData(self):
        self.xasixValue = []

        self.sigma_TT = np.append(self.sigma_TT, self.sigma_TT1)

        self.sigma_TT = np.append(self.sigma_TT, self.sigma_TT2)

        self.sigma_LT = np.append(self.sigma_LT, self.sigma_LT1)
        self.sigma_LT = np.append(self.sigma_LT, self.sigma_LT2)

        self.sigma_T = np.append(self.sigma_T, self.sigma_T1)
        self.sigma_T = np.append(self.sigma_T, self.sigma_T2)

        self.sigma_L = np.append(self.sigma_L, self.sigma_L1)
        self.sigma_L = np.append(self.sigma_L, self.sigma_L2)

        self.dsigma_TT = np.append(self.dsigma_TT, self.dsigma_TT1)
        self.dsigma_TT = np.append(self.dsigma_TT, self.dsigma_TT2)

        self.dsigma_LT = np.append(self.dsigma_LT, self.dsigma_LT1)
        self.dsigma_LT = np.append(self.dsigma_LT, self.dsigma_LT2)

        self.dsigma_T = np.append(self.dsigma_T, self.dsigma_T1)
        self.dsigma_T = np.append(self.dsigma_T, self.dsigma_T2)

        self.dsigma_L = np.append(self.dsigma_L, self.dsigma_L1)
        self.dsigma_L = np.append(self.dsigma_L, self.dsigma_L2)

        self.resA = np.append(self.resA, self.resA1)
        self.resA = np.append(self.resA, self.resA2)

        self.resC = np.append(self.resC, self.resC1)
        self.resC = np.append(self.resC, self.resC2)

        self.resB = np.append(self.resB, self.resB1)
        self.resB = np.append(self.resB, self.resB2)

        self.dresA = np.append(self.dresA, self.dresA1)
        self.dresA = np.append(self.dresA, self.dresA2)

        self.dresC = np.append(self.dresC, self.dresC1)
        self.dresC = np.append(self.dresC, self.dresC2)

        self.dresB = np.append(self.dresB, self.dresB1)
        self.dresB = np.append(self.dresB, self.dresB2)

        self.resCrossSect = np.append(self.resCrossSect, self.resCrossSect1)
        self.resCrossSect = np.append(self.resCrossSect, self.resCrossSect2)

        self.dresCrossSect = np.append(self.dresCrossSect, self.dresCrossSect1)
        self.dresCrossSect = np.append(self.dresCrossSect, self.dresCrossSect2)

        self.xasixValue = np.append(self.xasixValue, self.xasixValue1)
        self.xasixValue = np.append(self.xasixValue, self.xasixValue2)

        if (self.particle != 'Pi0P'):
            self.sigma_TT = np.append(self.sigma_TT, self.sigma_TT3)
            self.sigma_TT = np.append(self.sigma_TT, self.sigma_TT4)
            self.sigma_LT = np.append(self.sigma_LT, self.sigma_LT3)
            self.sigma_LT = np.append(self.sigma_LT, self.sigma_LT4)
            self.sigma_T = np.append(self.sigma_T, self.sigma_T3)
            self.sigma_T = np.append(self.sigma_T, self.sigma_T4)
            self.sigma_L = np.append(self.sigma_L, self.sigma_L3)
            self.sigma_L = np.append(self.sigma_L, self.sigma_L4)
            self.dsigma_TT = np.append(self.dsigma_TT, self.dsigma_TT3)
            self.dsigma_TT = np.append(self.dsigma_TT, self.dsigma_TT4)
            self.dsigma_LT = np.append(self.dsigma_LT, self.dsigma_LT3)
            self.dsigma_LT = np.append(self.dsigma_LT, self.dsigma_LT4)
            self.dsigma_T = np.append(self.dsigma_T, self.dsigma_T3)
            self.dsigma_T = np.append(self.dsigma_T, self.dsigma_T4)
            self.dsigma_L = np.append(self.dsigma_L, self.dsigma_L3)
            self.dsigma_L = np.append(self.dsigma_L, self.dsigma_L4)
            self.resA = np.append(self.resA, self.resA3)
            self.resA = np.append(self.resA, self.resA4)
            self.resC = np.append(self.resC, self.resC3)
            self.resC = np.append(self.resC, self.resC4)
            self.resB = np.append(self.resB, self.resB3)
            self.resB = np.append(self.resB, self.resB4)
            self.dresA = np.append(self.dresA, self.dresA3)
            self.dresA = np.append(self.dresA, self.dresA4)
            self.dresC = np.append(self.dresC, self.dresC3)
            self.dresC = np.append(self.dresC, self.dresC4)
            self.dresB = np.append(self.dresB, self.dresB3)
            self.dresB = np.append(self.dresB, self.dresB4)
            self.resCrossSect = np.append(self.resCrossSect, self.resCrossSect3)
            self.resCrossSect = np.append(self.resCrossSect, self.resCrossSect4)
            self.dresCrossSect = np.append(self.dresCrossSect, self.dresCrossSect3)
            self.dresCrossSect = np.append(self.dresCrossSect, self.dresCrossSect4)
            self.xasixValue = np.append(self.xasixValue, self.xasixValue3)
            self.xasixValue = np.append(self.xasixValue, self.xasixValue4)


try:
    graphObj = simpleMeasure(wValueUser=wValue, q2ValueUser=q2Value, cosValueUser=cosValue,
                             ebeamValueUser=eBeamValue, epsValueUser=epsValue, phiValueUser=phiValue,
                             particleUser=particle)
except:
    graphObj = simpleMeasure()

try:
    graphObj1 = simpleMeasure(wValueUser=wValue1, q2ValueUser=q2Value1, cosValueUser=cosValue1,
                              ebeamValueUser=eBeamValue1, epsValueUser=epsValue1, phiValueUser=phiValue1,
                              particleUser=particle1)
except:
    graphObj1 = simpleMeasure()

try:
    graphObj2 = simpleMeasure(wValueUser=wValue2, q2ValueUser=q2Value2, cosValueUser=cosValue2,
                              ebeamValueUser=eBeamValue2, epsValueUser=epsValue2, phiValueUser=phiValue2,
                              particleUser=particle2)
except:
    graphObj2 = simpleMeasure()

graphObj.findAndUseMethod()
graphObj.unionData()

graphObj1.findAndUseMethod()
graphObj1.unionData()

graphObj2.findAndUseMethod()
graphObj2.unionData()



def draw_graph(graphLabel,
               graphObj_xAxisValue=[],graphObj_xAxisValue1=[],graphObj_xAxisValue2=[],
               graphObj_value=[], graphObj_value1=[], graphObj_value2=[],
               dgraphObj_value=[], dgraphObj_value1=[], dgraphObj_value2=[],
               exp_data_xAxisValue=[],exp_data_xAxisValue1=[],exp_data_xAxisValue2=[],
               exp_data_value=[], exp_data_value1=[], exp_data_value2=[],
               dexp_data_value=[], dexp_data_value1=[], dexp_data_value2=[]):


    trace = go.Scatter(
        x=graphObj_xAxisValue,
        y=graphObj_value,
        marker_color='rgba(0, 0, 255, 0.3)',
        error_y=dict(
            type='data',
            array=dgraphObj_value,
            color='rgba(0, 0, 255, 0.3)',
            thickness=1.5,
            width=3),
        name='Interpolated data ' + name_0,
        marker_size=1)

    exp_trace = go.Scatter(
        x=exp_data_xAxisValue,
        y=exp_data_value,
        marker_color='rgba(0, 0, 255, 1)',
        mode='markers',
        error_y=dict(
            type='data',
            array=dexp_data_value,
            color='rgba(0, 0, 255, 1)',
            thickness=1.5,
            width=3),
        name='Experimental data ' + name_0,
        marker_size=1)

    trace1 = go.Scatter(
        x=graphObj_xAxisValue1,
        y=graphObj_value1,
        marker_color='rgba(0, 0, 0, 1)',
        error_y=dict(
            type='data',
            array=dgraphObj_value1,
            color='rgba(0, 0, 0, 1)',
            thickness=1.5,
            width=3),
        name='Interpolated data ' + name_1,
        marker_size=1)

    exp_trace1 = go.Scatter(
        x=exp_data_xAxisValue1,
        y=exp_data_value1,
        mode='markers',
        error_y=dict(
            type='data',
            array=dexp_data_value1,
            color='rgba(0, 0, 0, 1)',
            thickness=1.5,
            width=3),
        name='Experimental data ' + name_1,
        marker_size=1)

    trace2 = go.Scatter(
        x=graphObj_xAxisValue2,
        y=graphObj_value2,
        marker_color='rgba(255, 0, 0, 0.3)',
        error_y=dict(
            type='data',
            array=dgraphObj_value2,
            color='rgba(255, 0, 0, 0.3)',
            thickness=1.5,
            width=3),
        name='Interpolated data ' + name_2,
        marker_size=1)

    exp_trace2 = go.Scatter(
        x=exp_data_xAxisValue2,
        y=exp_data_value2,
        marker_color='rgba(255, 0, 0, 1)',
        mode='markers',
        error_y=dict(
            type='data',
            array=dexp_data_value2,
            color='rgba(255, 0, 0, 1)',
            thickness=1.5,
            width=3),
        name='Experimental data ' + name_2,
        marker_size=1)



    # data = [trace2,exp_trace2]
    # data = [trace,exp_trace,trace1,trace2,exp_trace2]
    data = [trace, exp_trace, trace1, exp_trace1, trace2, exp_trace2]

    fig = go.Figure(data=data)
    fig.layout.height = 700
    fig.layout.width = 1000
    fig.layout.title = graphLabel


    fig.layout.yaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.5,
        title=graphLabel,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))
    fig.layout.xaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.2,
        title=graphObj.xlabel,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))

    return fig

# fig.show()





print("Content-type: text/html\n")
print("""<!DOCTYPE HTML>
                <html>
                <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style type="text/css">
                        A { text-decoration: none;  color: red; } 
                        * { margin: 0;}
            .textBox { width: 1440px; height:80px; margin:auto; }
            .imagesBox{ width: 1440px; height:900px; margin:auto; }
            .textBox2 { width: 1440px; height:50px; margin:auto; }
            .tableBox1 {margin:auto;  width: 1440px; height:350px;}
            td { text-align: center ;}


                        </style>
                <meta charset="utf-8">
            <meta name="viewport" content="width=device-width">
                <script type="text/javascript"
                src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
                </script>                   
                <title>CLAS graph</title>
                    </head>
        <body>  
        <center>
        <h1>Interpolation</h1>
        <br> """)

print("<h1>{}</h1>".format(user_choose))



fig=draw_graph(graphObj_xAxisValue=graphObj.xasixValue, graphObj_xAxisValue1=graphObj1.xasixValue, graphObj_xAxisValue2=graphObj2.xasixValue,
           graphObj_value=graphObj.resCrossSect, graphObj_value1=graphObj1.resCrossSect, graphObj_value2=graphObj2.resCrossSect,
           dgraphObj_value=graphObj.dresCrossSect, dgraphObj_value1=graphObj1.dresCrossSect, dgraphObj_value2=graphObj2.dresCrossSect,
           exp_data_xAxisValue=exp_data['phi'], exp_data_xAxisValue1=exp_data1['phi'], exp_data_xAxisValue2=exp_data2['phi'],
           exp_data_value=exp_data['cross_section'], exp_data_value1=exp_data['cross_section'], exp_data_value2=exp_data2['cross_section'],
           dexp_data_value=exp_data['d_cross_section_all'],dexp_data_value1=exp_data1['d_cross_section_all'],dexp_data_value2=exp_data2['d_cross_section_all'],
           graphLabel='d\u03c3/d\u03a9(mcbn/sterad)')
print("{}".format(fig.to_html(full_html=False)))


fig=draw_graph(graphObj_xAxisValue=graphObj.xasixValue, graphObj_xAxisValue1=graphObj1.xasixValue, graphObj_xAxisValue2=graphObj2.xasixValue,
           graphObj_value=graphObj.resA, graphObj_value1=graphObj1.resA, graphObj_value2=graphObj2.resA,
           dgraphObj_value=graphObj.dresA, dgraphObj_value1=graphObj1.dresA, dgraphObj_value2=graphObj2.dresA,
          # exp_data_xAxisValue=exp_data['phi'], exp_data_xAxisValue1=exp_data1['phi'], exp_data_xAxisValue2=exp_data2['phi'],
          # exp_data_value=exp_data['cross_section'], exp_data_value1=exp_data['cross_section'], exp_data_value2=exp_data2['cross_section'],
           #dexp_data_value=exp_data['d_cross_section_all'],dexp_data_value1=exp_data1['d_cross_section_all'],dexp_data_value2=exp_data2['d_cross_section_all'],
           graphLabel='A(mcbn/sterad)')
print("{}".format(fig.to_html(full_html=False)))


fig=draw_graph(graphObj_xAxisValue=graphObj.xasixValue, graphObj_xAxisValue1=graphObj1.xasixValue, graphObj_xAxisValue2=graphObj2.xasixValue,
           graphObj_value=graphObj.resB, graphObj_value1=graphObj1.resB, graphObj_value2=graphObj2.resB,
           dgraphObj_value=graphObj.dresB, dgraphObj_value1=graphObj1.dresB, dgraphObj_value2=graphObj2.dresB,
          # exp_data_xAxisValue=exp_data['phi'], exp_data_xAxisValue1=exp_data1['phi'], exp_data_xAxisValue2=exp_data2['phi'],
          # exp_data_value=exp_data['cross_section'], exp_data_value1=exp_data['cross_section'], exp_data_value2=exp_data2['cross_section'],
           #dexp_data_value=exp_data['d_cross_section_all'],dexp_data_value1=exp_data1['d_cross_section_all'],dexp_data_value2=exp_data2['d_cross_section_all'],
           graphLabel='B(mcbn/sterad)')
print("{}".format(fig.to_html(full_html=False)))


fig=draw_graph(graphObj_xAxisValue=graphObj.xasixValue, graphObj_xAxisValue1=graphObj1.xasixValue, graphObj_xAxisValue2=graphObj2.xasixValue,
           graphObj_value=graphObj.resC, graphObj_value1=graphObj1.resC, graphObj_value2=graphObj2.resC,
           dgraphObj_value=graphObj.dresC, dgraphObj_value1=graphObj1.dresC, dgraphObj_value2=graphObj2.dresC,
          # exp_data_xAxisValue=exp_data['phi'], exp_data_xAxisValue1=exp_data1['phi'], exp_data_xAxisValue2=exp_data2['phi'],
          # exp_data_value=exp_data['cross_section'], exp_data_value1=exp_data['cross_section'], exp_data_value2=exp_data2['cross_section'],
           #dexp_data_value=exp_data['d_cross_section_all'],dexp_data_value1=exp_data1['d_cross_section_all'],dexp_data_value2=exp_data2['d_cross_section_all'],
           graphLabel='C(mcbn/sterad)')
print("{}".format(fig.to_html(full_html=False)))


print("""</center>
        </body>
         </html>""")
