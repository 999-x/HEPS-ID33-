import os, sys; sys.path.append(os.path.join('../../..', '..', '..'))  # analysis:ignore
import math
import numpy as np
# import matplotlib as mpl

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr
"毛细管材料的定义"
mGlass = rm.Material(('Si', 'O'), quantities=(1, 2), rho=2.2)

"从OE中引出BentCapillary,并定义相关参数，其中local_x0,local_x0Prime,local_r0,local_r0Prime是根据轴线系数方程计算的毛细管参数"
class BentCapillary(roe.OE):
    def __init__(self, *args, **kwargs):
        self.f1 = kwargs.pop('f1')
        self.entranceAlpha = kwargs.pop('entranceAlpha')
        self.f = kwargs.pop('f')
        self.x = kwargs.pop('x')
        self.r0in = kwargs.pop('rIn')
        self.r0out = kwargs.pop('rOut')
        roe.OE.__init__(self, *args, **kwargs)

        s0 = self.f - self.f1
        self.Rmax = 2.0
        self.s0 = s0
        self.isParametric = True

    def local_x0(self, s):  # axis of capillary, x(s)
        return self.x * (0.05725476 * (self.f - s - 15) - 0.00062746 * (self.f - s - 15)**2 + 1)

    def local_x0Prime(self, s):
        return self.x * (0.00125492 * (self.f-s-15) - 0.05725476)

    def local_r0(self, s):  # radius of capillary (s)
        return r0 * (0.05725476 * (self.f - s - 15) - 0.00062746 * (self.f - s - 15)**2 + 1)

    def local_r0Prime(self, s):
        return r0 * (0.00125492 * (self.f-s-15) - 0.05725476)

    def local_r(self, s, phi):
        den = np.cos(np.arctan(self.local_x0Prime(s)))**2
        return self.local_r0(s) / (np.cos(phi)**2/den + np.sin(phi)**2)

    def local_n(self, s, phi):
        a = -np.sin(phi)
        b = -np.sin(phi)*self.local_x0Prime(s) - self.local_r0Prime(s)
        c = -np.cos(phi)
        norm = np.sqrt(a**2 + b**2 + c**2)
        return a/norm, b/norm, c/norm

    def xyz_to_param(self, x, y, z):
        """ *s*, *r*, *phi* are cynindrc-like coordinates of the capillary.
        *s* is along y in the reverse direction, starting at the exit,
        *r* is measured from the capillary axis x0(s)
        *phi* is the polar angle measured from the z (vertical) direction."""
        s = self.f - y
        phi = np.arctan2(x - self.local_x0(s), z)
        r = np.sqrt((x-self.local_x0(s))**2 + z**2)
        return s, phi, r

    def param_to_xyz(self, s, phi, r):
        x = self.local_x0(s) + r*np.sin(phi)
        y = self.f - s
        z = r * np.cos(phi)
        return x, y, z

r"""
f1是毛细管的前焦距，f是f1和毛细管长度L的和，代表着从毛细管前焦距处到毛细管出口端的物理长度，r0是单根毛细管的管半径，wall是单根毛细管的壁厚，
layers是毛细管的层数，nRefl是最大反射次数，nRelfDisp和xzPrimeMax是用于polt图中的参数
以上参数都可修改
"""
f1 = 15
f = 81# f-f1=L
r0 = 0.005
wall = 0.00125
layers = 30  # number of hexagonal layers,30层，对应2611根单毛细管
nRefl = 12
nReflDisp = 12
xzPrimeMax = 7.

r"""
以下五行代码是罗兰圆中所用晶体的材料定义，以及是否启动TT方程
"""
useTT = True
crystalMaterial = 'Si'
a = 5.4305
crystal = rm.CrystalDiamond((6, 6, 0), a/math.sqrt(72), t=0.35, useTT=useTT)
nprocesses = 4

r"""
罗兰圆半径R，以及分析晶体的尺寸
"""
R = 500.  # mm
dxCrystal = 100.
dyCrystal = 100.

r"""
根据xrtBentXtal计算的9689eV对应的Si(6,6,0)的布拉格角，以及斜切角
根据该角度精确计算布拉格衍射的中心能量，考虑了晶体折射效应导致的微小角度偏移。 
E0raw是初步根据布拉格定律计算的能量，而E0是考虑了折射效应（对称布拉格衍射的角偏移）后的更精确的中心能量。
这个偏移量通常很小，但对于高精度计算是必要的
"""
thetaDegree = 88.586,
alphaDegree = 0.,
theta = np.radians(thetaDegree)
sinTheta = np.sin(theta)
E0raw = rm.ch / (2 * crystal.d * sinTheta)
dTheta = crystal.get_dtheta_symmetric_Bragg(E0raw)
E0 = rm.ch / (2 * crystal.d * math.sin(theta + dTheta))
offsetE = round(E0, 3)
eAxisNormal = 1e-4 #能量展宽系数，根据实验经验设置
eAxisMin = E0 * (1. - eAxisNormal)
eAxisMax = E0 * (1. + eAxisNormal)

r"""
根据布拉格角计算吻合罗兰圆的分析晶体以及探测器的坐标
为后续光源刚好覆盖分析晶体的发散角，以及分析晶体和探测器坐标的设置做好准备工作
"""
alpha1 = np.radians(alphaDegree)
p = 2. * R * math.sin(theta + alpha1)
q = 2. * R * math.sin(theta - alpha1)
sin2Theta = math.sin(2 * theta)
cos2Theta = math.cos(2 * theta)
Rs = 2. * R * sinTheta**2
yDet = p + q * cos2Theta
zDet = q * sin2Theta
pdp = 2. * R * math.sin(theta + alpha1 - dyCrystal/6/R)

r"""
对光源、多毛细管透镜、分析晶体以及探测器的定义
"""
def build_beamline(nrays=2000):
    beamLine = raycing.BeamLine(azimuth=0, height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', nrays=nrays, dx=0.005, dy=0,
        dz=0.005, distxprime='annulus', dxprime=0.05, distzprime='normal', dzprime=0.05,
        distE='normal', energies=(E0, 1), polarization='horizontal')
#    beamLine.sources[0].dxprime = dxCrystal / pdp
#    beamLine.sources[0].dzprime = dyCrystal * \
#                                  math.sin(theta + alpha1) / pdp

#   ===========================================================
    beamLine.fsm1 = rsc.Screen(beamLine, 'DiamondFSM1', (0, f1, 0))
    beamLine.capillaries = []
    beamLine.firstInLayer = []
    beamLine.xzMax = 0
    for n in range(layers):
        if n > 0:
            ms = range(n)
            i6 = range(6)
        else:
            ms = 0,
            i6 = 0,
        beamLine.firstInLayer.append(len(beamLine.capillaries))
        for i in i6:
            for m in ms:
                x = 2 * (r0+wall) * (n**2 + m**2 - n*m)**0.5
                alpha = np.arcsin(x / f1)
                roll1 = -np.arctan2(np.sqrt(3)*m, 2*n - m)
                roll = roll1 + i*np.pi/3.
                capillary = BentCapillary(
                    beamLine, 'BentCapillary', [0, 0, 0], roll=roll,
                    material=mGlass, limPhysY=[f1, f],#y的范围决定s的范围
                    #yaw=0.003,
                    f=f, f1=f1, x=x, entranceAlpha=alpha, rIn=r0, rOut=1.25 * r0)
                beamLine.capillaries.append(capillary)
                if beamLine.xzMax < capillary.Rmax:
                    beamLine.xzMax = capillary.Rmax
    print('max divergence =', alpha)
    beamLine.xzMax += 2 * r0
    print(len(beamLine.capillaries))
    beamLine.sources[0].dxprime = 0, np.arcsin((2*n+1) * (r0+wall) / (f1))
    beamLine.fsm2 = rsc.Screen(beamLine, 'DiamondFSM2', (0, f+64, 0))#f=81，后焦距为64mm
#   ============================================================毛细管部分

    beamLine.analyzer = roe.JohannToroid(
        beamLine, 'JohannAnalyzer', surface=('',),
        limPhysX=(-dxCrystal/2, dxCrystal/2),
        limPhysY=(-dyCrystal/2, dyCrystal/2),
        Rm=1000, shape='round',
        targetOpenCL='auto' if useTT else None, precisionOpenCL='float32')
    beamLine.analyzer.alpha = alpha
    beamLine.analyzer.center = 0, p+69.25-0, 0
    beamLine.analyzer.pitch = theta + alpha
    beamLine.analyzer.Rs = Rs
    r"""
    如果 off-rowland, p+69.25
    如果 off-rowland, yDet+130, zDet-3
    detector的x，z的设置是为了让其对准分析晶体
    """
    beamLine.detector = rsc.Screen(beamLine, 'Detector', x=(1, 0, 0))
    beamLine.detector.center = 0, yDet+130, zDet-3
    beamLine.detector.z = 0, -sin2Theta, cos2Theta
    return beamLine

r"""
模拟中光束的传输过程里，按传输顺序：光源→多毛细管透镜→分析晶体→探测器
分析晶体所反射的光束是经过多毛细管透镜后的光束，探测器所暴露的光线是经过分析晶体反射的光束
"""
def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()

    beamFSM1 = beamLine.fsm1.expose(beamSource)
    outDict = {'beamSource': beamSource, 'beamFSM1': beamFSM1}
    beamCapillaryGlobalTotal = None
    for i, capillary in enumerate(beamLine.capillaries):
        beamCapillaryGlobal, beamCapillaryLocalN =\
            capillary.multiple_reflect(beamSource, maxReflections=nRefl)
        beamCapillaryLocalN.phi /= np.pi
        if beamCapillaryGlobalTotal is None:
            beamCapillaryGlobalTotal = beamCapillaryGlobal
        else:
            good = ((beamCapillaryGlobal.state == 1) |
                    (beamCapillaryGlobal.state == 3))
            rs.copy_beam(beamCapillaryGlobalTotal, beamCapillaryGlobal,
                         good, includeState=True)
        outDict['beamCapillaryLocalN{0:02d}'.format(i)] = beamCapillaryLocalN
    outDict['beamCapillaryGlobalTotal'] = beamCapillaryGlobalTotal
    beamFSM2 = beamLine.fsm2.expose(beamCapillaryGlobalTotal)
    outDict['beamFSM2'] = beamFSM2

    beamAnalyzerGlobal, beamAnalyzerLocal = \
        beamLine.analyzer.reflect(beamCapillaryGlobalTotal)
    beamDetector = beamLine.detector.expose(beamAnalyzerGlobal)
    outDict.update({'beamAnalyzerGlobal': beamAnalyzerGlobal,
                    'beamAnalyzerLocal': beamAnalyzerLocal,
                    'beamDetector': beamDetector})
    return outDict
rr.run_process = run_process

r"""
接下来就是绘图模块，需要观察那个部位的光束情况，添加上即可
"""
def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSM1', (1, 3, -1),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=256, ppb=2),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=256, ppb=2),
        caxis='category', beamState='beamFSM2', title='FSM1_Cat')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM2', (1, 3),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', bins=256, ppb=2),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=256, ppb=2),
        caxis='category', title='FSM2_xzCat')
    plot.xaxis.fwhmFormatStr = fwhmFormatStrE
    plot.yaxis.fwhmFormatStr = fwhmFormatStrE
    plot.xaxis.limits = [-1, 1]#[-beamLine.xzMax, beamLine.xzMax]
    plot.yaxis.limits = [-1, 1]#[-beamLine.xzMax, beamLine.xzMax]
    #    plot.fluxFormatStr = '%.2e'
    plots.append(plot)

    plotAnE = xrtp.XYCPlot(
        'beamAnalyzerLocal', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-dxCrystal / 2, dxCrystal / 2], bins=400, ppb=1),
        yaxis=xrtp.XYCAxis(r'$y$', 'mm', limits=[-dyCrystal / 2, dyCrystal / 2], bins=400, ppb=1),
        caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f',
                           bins=200, ppb=2),
        title='xtal_E', oe=beamLine.analyzer)
    plotAnE.caxis.offset = offsetE
    plotAnE.caxis.limits = [eAxisMin, eAxisMax]
    plotAnE.caxis.fwhmFormatStr = fwhmFormatStrE
    plotAnE.caxis.invertAxis = True
    plotAnE.textPanel = plotAnE.fig.text(
        0.88, 0.85, '', transform=plotAnE.fig.transFigure, size=14, color='r',
        ha='center')
    plotAnE.saveName = 'xtal_E' + '.png'
    plots.append(plotAnE)

    plotDetE = xrtp.XYCPlot(
        'beamDetector', (1,), aspect='equal',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-7, 7],
                           fwhmFormatStr='%.3f'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-7, 7],
                           fwhmFormatStr='%.3f'),
        title='det_E')
    plotDetE.caxis.offset = offsetE
    plotDetE.caxis.limits = [eAxisMin, eAxisMax]
    plotDetE.caxis.fwhmFormatStr = fwhmFormatStrE
    plotDetE.caxis.invertAxis = True
    plotDetE.textPanel = plotDetE.fig.text(
        0.88, 0.8, '', transform=plotDetE.fig.transFigure, size=14, color='r',
        ha='center')
    plotDetE.saveName = 'det_E' + '.png'
    plots.append(plotDetE)

    return plots

r"""
最后是主函数，包含前面的几个模块。
值得注意的是，由于光线经过多毛细管透镜时，需要对每根毛细管都进行全外反射的计算，计算量大，如果一次模拟中设置的光线数量较大，会爆内存，报错
因此采取适量光线数，通过repeats多次重复模拟累积，来解决该问题。
这一问题实际受限于运行代码所用的设备，若硬件设备好，则可能无需面对该问题
"""
def main():
    beamLine = build_beamline()
    plots = define_plots(beamLine)
    xrtr.run_ray_tracing(
        plots,beamLine=beamLine, repeats=100, processes=1 if useTT else nprocesses)


if __name__ == '__main__':
    main()

