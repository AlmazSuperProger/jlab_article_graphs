#!/usr/bin/env python3
# -*- coding: utf-8 -*
from __future__ import unicode_literals

import cgi
import re
import base64
import sys
import re
import os.path
import csv
import math
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
from plotly.graph_objs import Scatter

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
from scipy.interpolate import griddata

# считываем данные, введенные пользователем


gettext = cgi.FieldStorage()
q2Value = gettext.getfirst("q2", "empty")
wValue = gettext.getfirst("w", "empty")
cosValue = gettext.getfirst("cos", "empty")
phiValue = gettext.getfirst("phi", "empty")
eBeamValue = gettext.getfirst("eBeam", "empty")
particle = gettext.getfirst("particle", "empty")
epsValue = gettext.getfirst("eps", "empty")

q2Value1 = gettext.getfirst("q21", "empty")
wValue1 = gettext.getfirst("w1", "empty")
cosValue1 = gettext.getfirst("cos1", "empty")
phiValue1 = gettext.getfirst("phi1", "empty")
eBeamValue1 = gettext.getfirst("eBeam1", "empty")
particle1 = gettext.getfirst("particle1", "empty")
epsValue1 = gettext.getfirst("eps1", "empty")

df = pd.read_csv('final_table.csv', header=None, sep='\t',
                 names=['Channel', 'MID',
                        'Wmin', 'Wmax', 'Q2min', 'Q2max', 'Cos(theta)',
                        'Sigma_T', 'dSigma_T', 'Sigma_L', 'dSigma_L',
                        'Sigma_TT', 'dSigma_TT', 'Sigma_LT', 'dSigma_LT',
                        'eps'])

a1new = (df['Wmin'] + df['Wmax']) / 2
a2new = (df['Q2min'] + df['Q2max']) / 2
df = df.assign(Waverage=a1new)
df = df.assign(Q2average=a2new)

pointsNum = 100
mp = 0.93827


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
        # Ввели WQ/QCOS/WCOS и phi  - method 22/32/42
        # строим зависимость структурных функций (TT, TL) от переменной Default
        # Ввели WQ/QCOS/WCOS и ebeam и phi  - method 23/33/43
        # строим зависимость сечения и структурных функций (A, TT, TL) от переменной Default


def onePlotlyGraphWithErrors(xarray, yarray, layoutTitle, xLabel, errosArr):
    trace = go.Scatter(
        x=xarray,
        y=yarray,
        error_y=dict(
            type='data',
            array=errosArr,
            color='rgba(100, 100, 255, 0.6)',
            thickness=1.5,
            width=3),
        name='Interpolation',
        marker_size=1)
    data = [trace]

    fig = go.Figure(data=data)
    fig.layout.height = 700
    fig.layout.width = 1000
    fig.layout.title = layoutTitle

    fig.layout.yaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.5,
        title=layoutTitle,
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
        title=xLabel,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))

    return fig


def twoPlotlyGraphWithErrors(xarray, yarray, xarray1, yarray1, layoutTitle, xLabel, errosArr, errosArr1):
    trace = go.Scatter(
        x=xarray,
        y=yarray,
        error_y=dict(
            type='data',
            array=errosArr,
            color='rgba(100, 100, 255, 0.6)',
            thickness=1.5,
            width=3),
        name='Interpolation',
        marker_size=1)
    trace1 = go.Scatter(
        x=xarray1,
        y=yarray1,
        mode='markers',
        error_y=dict(
            type='data',
            array=errosArr1,
            color='rgba(255, 100, 100, 1)',
            thickness=1.5,
            width=3),
        name='Interpolation data',
        marker_size=3)

    data = [trace, trace1]

    fig = go.Figure(data=data)
    fig.layout.height = 700
    fig.layout.width = 1000
    fig.layout.title = layoutTitle

    fig.layout.yaxis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=1,
        zerolinecolor='black',
        zerolinewidth=0.5,
        linewidth=0.5,
        title=layoutTitle,
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
        title=xLabel,
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='black'
        ))

    return fig


try:
    graphObj = simpleMeasure(wValueUser=wValue, q2ValueUser=q2Value, cosValueUser=cosValue, ebeamValueUser=eBeamValue,
                             epsValueUser=epsValue, phiValueUser=phiValue, particleUser=particle)
except:
    graphObj = simpleMeasure()

try:
    graphObj1 = simpleMeasure(wValueUser=wValue1, q2ValueUser=q2Value1, cosValueUser=cosValue1,
                              ebeamValueUser=eBeamValue1, epsValueUser=epsValue1, phiValueUser=phiValue1,
                              particleUser=particle1)
except:
    graphObj1 = simpleMeasure()

graphObj.findAndUseMethod()
graphObj.unionData()

graphObj1.findAndUseMethod()
graphObj1.unionData()

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
        <br> 







        <a href="https://clas.sinp.msu.ru/cgi-bin/almaz/instruction">Instruction and available data areas</a>
        <br>
        <br>
        <br>

        <form method="GET" action="https://clas.sinp.msu.ru/cgi-bin/almaz/interpolateGraph.py" >

      <p> <input  type="text" name="eBeam"  placeholder="Ebeam(GeV)"> </p>
      <input  type="text" name="w"  placeholder="W(GeV)" >
          <input  type="text" name="q2"  placeholder="Q2(GeV2)" >
      <input  type="text" name="eps"  placeholder="eps">
          <input  type="text" name="cos"  placeholder="Cos(theta)" >
          <input  type="text" name="phi"  placeholder="phi(degree)">
          <br>
          <br>
            <select class="select" name="particle" size="1">
            <option value="Pin">gvp--->ПЂвЃєn</option>
            <option value="Pi0P">gvp--->ПЂвЃ°p</option>
            </select>
          <br>
          <br>
        If you want to compare two experiments, then fill in the same columns in the bottom input field
           <br>
        If you do not want to compare two experiments, then leave the bottom input field blank
        <br>
          <br>
          <p> <input  type="text" name="eBeam1"  placeholder="Ebeam(GeV)"> </p>
          <input  type="text" name="w1"  placeholder="W(GeV)" >
          <input  type="text" name="q21"  placeholder="Q2(GeV2)" >
          <input  type="text" name="eps1"  placeholder="eps">
          <input  type="text" name="cos1"  placeholder="Cos(theta)" >
          <input  type="text" name="phi1"  placeholder="phi(degree)">
          <br>
          <br>
            <select class="select" name="particle1" size="1">
            <option value="Pin">gvp--->ПЂвЃєn</option>
            <option value="Pi0P">gvp--->ПЂвЃ°p</option>
            </select>
          <br>
          <br>

          <br>
          <br>
         <p> <input class="button" class="submitbutton" type="submit" value="Find">  </p>
         <br>
        </form>""")

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
# строим зависимость структурных функций (A, TT, TL) от переменной Default
# Ввели WQ/QCOS/WCOS и phi  - method 22/32/42
# строим зависимость структурных функций (TT, TL) от переменной Default
# Ввели WQ/QCOS/WCOS и ebeam и phi  - method 23/33/43
# строим зависимость сечения и структурных функций (A, TT, TL) от переменной Default


if ((graphObj.method != 0) and (graphObj1.method == 0)):
    graphObj.getEnteredDataOneTable()

    if (graphObj.method == 1):
        graphObj.getTT_TL_T_L_tableOne()
    if (graphObj.method == 11):
        fig = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resCrossSect, 'd\u03c3/d\u03a9(mcbn/sterad)',
                                       graphObj.xlabel, graphObj.dresCrossSect)
        print("{}".format(fig.to_html(full_html=False, include_plotlyjs=False)))
    if (graphObj.method == 12):
        graphObj.getTT_TL_T_L_tableOne()
    if (graphObj.method == 13):
        graphObj.getCross_resA_TT_LT_tableOne()

    if ((graphObj.method == 2) or (graphObj.method == 3) or (graphObj.method == 4)):
        fig2 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_LT)
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj.method == 21) or (graphObj.method == 31) or (graphObj.method == 41)):
        fig1 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resA, 'd\u03c3<sub>u</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dresA)
        fig2 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_LT)
        print("{}".format(fig1.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj.method == 22) or (graphObj.method == 32) or (graphObj.method == 42)):
        fig2 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_LT)
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj.method == 23) or (graphObj.method == 33) or (graphObj.method == 43)):
        fig = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resCrossSect, 'd\u03c3/d\u03a9(mcbn/sterad)',
                                       graphObj.xlabel, graphObj.dresCrossSect)
        fig1 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resA, 'd\u03c3<sub>u</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dresA)
        fig2 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                        graphObj.dsigma_LT)
        fig4 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_T,
                                        'd\u03c3<sub>T</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel, graphObj.dsigma_T)
        fig5 = onePlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_L,
                                        'd\u03c3<sub>L</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel, graphObj.dsigma_L)
        print("{}".format(fig.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig1.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig4.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig5.to_html(full_html=False, include_plotlyjs=False)))

if ((graphObj.method == 0) and (graphObj1.method != 0)):
    graphObj1.getEnteredDataOneTable()

    if (graphObj1.method == 1):
        graphObj1.getTT_TL_T_L_tableOne()
    if (graphObj1.method == 11):
        fig = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.resCrossSect, 'd\u03c3/d\u03a9(mcbn/sterad)',
                                       graphObj1.xlabel, graphObj1.dresCrossSect)
        print("{}".format(fig.to_html(full_html=False, include_plotlyjs=False)))
    if (graphObj1.method == 12):
        graphObj1.getTT_TL_T_L_tableOne()
    if (graphObj1.method == 13):
        graphObj1.getCross_resA_TT_LT_tableOne()

    if ((graphObj1.method == 2) or (graphObj1.method == 3) or (graphObj1.method == 4)):
        fig2 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_LT)
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj1.method == 21) or (graphObj1.method == 31) or (graphObj1.method == 41)):
        fig1 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.resA,
                                        'd\u03c3<sub>u</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel, graphObj1.dresA)
        fig2 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_LT)
        print("{}".format(fig1.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj1.method == 22) or (graphObj1.method == 32) or (graphObj1.method == 42)):
        fig2 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_LT)
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj1.method == 23) or (graphObj1.method == 33) or (graphObj1.method == 43)):
        fig = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.resCrossSect, 'd\u03c3/d\u03a9(mcbn/sterad)',
                                       graphObj1.xlabel, graphObj1.dresCrossSect)
        fig1 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.resA,
                                        'd\u03c3<sub>u</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel, graphObj1.dresA)
        fig2 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_TT,
                                        'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_TT)
        fig3 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_LT,
                                        'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_LT)
        fig4 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_T,
                                        'd\u03c3<sub>T</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_T)
        fig5 = onePlotlyGraphWithErrors(graphObj1.xasixValue, graphObj1.sigma_L,
                                        'd\u03c3<sub>L</sub>/d\u03a9(mcbn/sterad)', graphObj1.xlabel,
                                        graphObj1.dsigma_L)
        print("{}".format(fig.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig1.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig4.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig5.to_html(full_html=False, include_plotlyjs=False)))

# ДВА ГРАФИКА НА ОДНОМ РИСУНКЕ


if (graphObj.method == graphObj1.method and (graphObj.method != 0)):
    graphObj.getEnteredDataTwoTable(secondObject=graphObj1)
    if (graphObj.method == 1):
        graphObj.getTT_TL_T_L_tableTwo(objc=graphObj1)
    if (graphObj.method == 11):
        fig = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resCrossSect, graphObj1.xasixValue,
                                       graphObj1.resCrossSect, 'd\u03c3/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                       graphObj.dresCrossSect, graphObj1.dresCrossSect)
        print("{}".format(fig.to_html(full_html=False, include_plotlyjs=False)))
    if (graphObj.method == 12):
        graphObj.getTT_TL_T_L_tableTwo(objc=graphObj1)
    if (graphObj.method == 13):
        graphObj.getCross_resA_TT_LT_tableTwo(objc=graphObj1)

    if ((graphObj.method == 2) or (graphObj.method == 3) or (graphObj.method == 4)):
        fig2 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT, graphObj1.xasixValue,
                                        graphObj1.sigma_TT, 'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_TT, graphObj1.dsigma_TT)
        fig3 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT, graphObj1.xasixValue,
                                        graphObj1.sigma_LT, 'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_LT, graphObj1.dsigma_LT)
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj.method == 21) or (graphObj.method == 31) or (graphObj.method == 41)):
        fig1 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resA, graphObj1.xasixValue, graphObj1.resA,
                                        'd\u03c3<sub>u</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel, graphObj.dresA,
                                        graphObj1.dresA)
        fig2 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT, graphObj1.xasixValue,
                                        graphObj1.sigma_TT, 'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_TT, graphObj1.dsigma_TT)
        fig3 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT, graphObj1.xasixValue,
                                        graphObj1.sigma_LT, 'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_LT, graphObj1.dsigma_LT)
        print("{}".format(fig1.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj.method == 22) or (graphObj.method == 32) or (graphObj.method == 42)):
        fig2 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT, graphObj1.xasixValue,
                                        graphObj1.sigma_TT, 'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_TT, graphObj1.dsigma_TT)
        fig3 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT, graphObj1.xasixValue,
                                        graphObj1.sigma_LT, 'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_LT, graphObj1.dsigma_LT)
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))
    if ((graphObj.method == 23) or (graphObj.method == 33) or (graphObj.method == 43)):
        fig = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resCrossSect, graphObj1.xasixValue,
                                       graphObj1.resCrossSect, 'd\u03c3/d\u03a9(mcbn/sterad)', graphObj.xlabel,
                                       graphObj.dresCrossSect, graphObj1.dresCrossSect)
        fig1 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.resA, graphObj1.xasixValue, graphObj1.resA,
                                        'd\u03c3<sub>u</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel, graphObj.dresA,
                                        graphObj1.dresA)
        fig2 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_TT, graphObj1.xasixValue,
                                        graphObj1.sigma_TT, 'd\u03c3<sub>TT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_TT, graphObj1.dsigma_TT)
        fig3 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_LT, graphObj1.xasixValue,
                                        graphObj1.sigma_LT, 'd\u03c3<sub>LT</sub>/d\u03a9(mcbn/sterad)',
                                        graphObj.xlabel, graphObj.dsigma_LT, graphObj1.dsigma_LT)
        fig4 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_T, graphObj1.xasixValue, graphObj1.sigma_T,
                                        'd\u03c3<sub>T</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel, graphObj.dsigma_T,
                                        graphObj1.dsigma_T)
        fig5 = twoPlotlyGraphWithErrors(graphObj.xasixValue, graphObj.sigma_L, graphObj1.xasixValue, graphObj1.sigma_L,
                                        'd\u03c3<sub>L</sub>/d\u03a9(mcbn/sterad)', graphObj.xlabel, graphObj.dsigma_L,
                                        graphObj1.dsigma_L)
        print("{}".format(fig.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig1.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig2.to_html(full_html=False, include_plotlyjs=False)))
        print("{}".format(fig3.to_html(full_html=False, include_plotlyjs=False)))

if (graphObj.method != graphObj1.method and (graphObj.method != 0) and (graphObj1.method != 0)):
    print(" please fill the same columns or leave the bottom input field blank ")

if (graphObj.method == graphObj1.method and (graphObj.method == 0)):
    print(" please read the instruction ")

print("""</center>
        </body>
         </html>""")
