import os, sys

sys.path.append(os.path.join('../../..', '..', '..'))  # analysis:ignore
import math
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

import xrt.plotter as xrtp
import xrt.runner as xrtr

useTT = True
crystalMaterial = 'Si'
a = 5.4305

crystal = rm.CrystalDiamond((6, 6, 0), a / math.sqrt(72), t=0.35, useTT=useTT)

nprocesses = 4

R = 500.  # mm

dxCrystal = 100.
dyCrystal = 100.  # 分析晶体尺寸

thetaDegree = 88.586
alphaDegree = 0.

theta = np.radians(thetaDegree)
sinTheta = np.sin(theta)
E0raw = rm.ch / (2 * crystal.d * sinTheta)
dTheta = crystal.get_dtheta_symmetric_Bragg(E0raw)
E0 = rm.ch / (2 * crystal.d * math.sin(theta + dTheta))
offsetE = round(E0, 3)
eAxisNormal = 5e-5  # 能量展宽系数，根据实验经验设置
eAxisMin = E0 * (1. - eAxisNormal)
eAxisMax = E0 * (1. + eAxisNormal)

alpha = np.radians(alphaDegree)
p = 2. * R * math.sin(theta + alpha)
p1 = p + 69.25
q = 2. * R * math.sin(theta - alpha)
sin2Theta = math.sin(2 * theta)
cos2Theta = math.cos(2 * theta)
Rs = 2. * R * sinTheta ** 2
yDet = p + q * cos2Theta
zDet = q * sin2Theta
pdp = 2. * R * math.sin(theta + alpha - dyCrystal / 6 / R)

# ============ 分析晶体参数配置表 ============
# 存储每个晶体的独立参数配置
analyzer_configs = {}


def calculate_default_params(i, j):
    """计算晶体的默认参数"""
    gap_degree = 0.1222  # 7°
    theta_i = (2 - i) * gap_degree
    theta_j = (1 - j) * gap_degree

    default_center = (p1 * math.sin(theta_i) * math.cos(theta_j),
                      p1 * math.cos(theta_i) * math.cos(theta_j),
                      p1 * math.sin(theta_j))
    default_pitch = theta + alpha + (1 - j) * 0.129
    default_yaw = (i - 2) * 0.13

    return default_center, default_pitch, default_yaw


def update_analyzer_config(i, j, center=None, pitch=None, yaw=None):
    """更新单个分析晶体的参数 - 必须在 build_beamline() 之前调用"""
    if (i, j) not in analyzer_configs:
        # 如果配置不存在，先计算默认值
        default_center, default_pitch, default_yaw = calculate_default_params(i, j)
        analyzer_configs[(i, j)] = {
            "center": default_center,
            "pitch": default_pitch,
            "yaw": default_yaw,
            "modified": False  # 标记是否被修改过
        }

    # 更新参数并标记为已修改
    config = analyzer_configs[(i, j)]
    if center is not None:
        config["center"] = center
        config["modified"] = True
    if pitch is not None:
        config["pitch"] = pitch
        config["modified"] = True
    if yaw is not None:
        config["yaw"] = yaw
        config["modified"] = True


def get_analyzer_params(i, j):
    """获取晶体参数，如果配置表中没有则计算默认值"""
    if (i, j) not in analyzer_configs:
        # 如果配置不存在，计算默认值并保存
        center, pitch, yaw = calculate_default_params(i, j)
        analyzer_configs[(i, j)] = {
            "center": center,
            "pitch": pitch,
            "yaw": yaw,
            "modified": False
        }

    config = analyzer_configs[(i, j)]
    return config["center"], config["pitch"], config["yaw"], config.get("modified", False)


def build_beamline(nrays=1e6):
    beamLine = raycing.BeamLine(azimuth=0, height=0)
    rs.GeometricSource(
        beamLine, 'GeometricSource', center=[0, 0, 0], nrays=nrays, dx=0.005, dy=0,
        dz=0.005, distxprime='normal', dxprime=0.05, distzprime='normal', dzprime=0.05,
        distE='normal', energies=(E0, 1), polarization='horizontal')
    # 扩大水平和垂直方向发散角以覆盖15个晶体
    beamLine.sources[0].dxprime = dxCrystal * 1.5 / pdp
    beamLine.sources[0].dzprime = dyCrystal * math.sin(theta + alpha) * 1.5 / pdp
    beamLine.analyzers = []  # 存储所有分析晶体的列表

    # 创建15个（5×3）JohannToroid晶体
    beamLine.analyzers = []
    for i in range(5):
        for j in range(3):
            # 获取晶体参数（从配置表或使用默认值）
            center, pitch, yaw, modified = get_analyzer_params(i, j)

            analyzer = roe.JohannToroid(
                beamLine, f'JohannAnalyzer_{i}_{j}', surface=('',),
                limPhysX=(-dxCrystal / 2, dxCrystal / 2),
                limPhysY=(-dyCrystal / 2, dyCrystal / 2),
                Rm=1000, shape='round',
                targetOpenCL='auto' if useTT else None, precisionOpenCL='float32')
            analyzer.Rs = Rs
            analyzer.alpha = alpha
            analyzer.center = center
            analyzer.pitch = pitch
            analyzer.yaw = yaw
            # 添加网格索引信息
            analyzer.grid_i = i  # 列索引
            analyzer.grid_j = j  # 行索引
            analyzer.modified = modified  # 标记是否被修改过

            beamLine.analyzers.append(analyzer)

    """
    探测器的面，对准分析晶体，我们从分析晶体看向探测器，x轴正向朝右，z轴正向朝上发。 探测器的局部坐标轴与整个全局坐标轴相比，x轴反向
    """
    beamLine.detector = rsc.Screen(beamLine, 'Detector', x=(-1, 0, 0))
    beamLine.detector.center = 0, yDet + 130, zDet - 3
    beamLine.detector.z = 0, sin2Theta, -cos2Theta
    return beamLine


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()

    # 创建字典来存储所有分析器的结果
    outDict = {'beamSource': beamSource}

    # 处理每个分析器
    for idx, analyzer in enumerate(beamLine.analyzers):
        i = analyzer.grid_i
        j = analyzer.grid_j

        # 反射光束
        beamAnalyzerGlobal, beamAnalyzerLocal = analyzer.reflect(beamSource)

        # 曝光探测器
        beamDetector = beamLine.detector.expose(beamAnalyzerGlobal)

        # 为每个分析器分别存储结果
        outDict[f'beamAnalyzerGlobal_{i}_{j}'] = beamAnalyzerGlobal
        outDict[f'beamAnalyzerLocal_{i}_{j}'] = beamAnalyzerLocal
        outDict[f'beamDetector_{i}_{j}'] = beamDetector

        # 将最后一个分析器的结果作为默认（保持向后兼容）
        if idx == len(beamLine.analyzers) - 1:
            outDict['beamAnalyzerGlobal'] = beamAnalyzerGlobal
            outDict['beamAnalyzerLocal'] = beamAnalyzerLocal
            outDict['beamDetector'] = beamDetector

    return outDict


rr.run_process = run_process


def define_plots(beamLine):
    fwhmFormatStrE = '%.2f'
    plots = []

    # 1. 绘制所有分析器在探测器上的光斑（分开显示），并保存.mat文件
    print("\nCreating detector spot plots for each analyzer...")
    for i in range(5):
        for j in range(3):
            plotDet = xrtp.XYCPlot(
                f'beamDetector_{i}_{j}', (1,), aspect='equal',
                xaxis=xrtp.XYCAxis(r'$x$', 'mm', limits=[-7, 7],
                                   fwhmFormatStr='%.3f'),
                yaxis=xrtp.XYCAxis(r'$z$', 'mm', limits=[-7, 7],
                                   fwhmFormatStr='%.3f'),
                caxis=xrtp.XYCAxis('energy', 'eV', fwhmFormatStr='%.2f',
                                   bins=200, ppb=2),
                title=f'Detector Spot - Analyzer ({i},{j})',
                saveName=f'detector_{i}_{j}.png',
                persistentName=f'detector_{i}_{j}.mat')  # 保存为.mat文件
            plotDet.caxis.offset = offsetE
            plotDet.caxis.limits = [eAxisMin, eAxisMax]
            plotDet.caxis.fwhmFormatStr = fwhmFormatStrE
            plotDet.caxis.invertAxis = True
            plots.append(plotDet)
            print(f"  Created detector spot plot for analyzer ({i},{j})")
            print(f"  Will save histogram data to: detector_{i}_{j}.mat")

    # 2. 绘制每个分析器上的能量分布图（在分析晶体局部坐标系中）
    print("\nCreating energy distribution plots on analyzers for each analyzer...")
    for i in range(5):
        for j in range(3):
            plotEnergy = xrtp.XYCPlot(
                f'beamAnalyzerLocal_{i}_{j}', (1,), aspect='equal',
                xaxis=xrtp.XYCAxis(r'$x_{local}$', 'mm',
                                   limits=[-dxCrystal / 2, dxCrystal / 2]),
                yaxis=xrtp.XYCAxis(r'$y_{local}$', 'mm',
                                   limits=[-dyCrystal / 2, dyCrystal / 2]),
                caxis=xrtp.XYCAxis(r'$E$', 'eV', fwhmFormatStr=fwhmFormatStrE,
                                   data=raycing.get_energy),
                title=f'Energy on Analyzer ({i},{j})',
                saveName=f'energy_analyzer_{i}_{j}.png')
            plotEnergy.caxis.offset = offsetE
            plotEnergy.caxis.limits = [eAxisMin, eAxisMax]
            plots.append(plotEnergy)
            print(f"  Created energy distribution plot on analyzer ({i},{j})")

    print(f"\nTotal plots created: {len(plots)}")
    print("  - 15 detector spot plots (one for each analyzer)")
    print("  - 15 energy distribution plots on analyzers (one for each analyzer)")
    print("\nNote: Each plot will save histogram data to a .mat file for MATLAB processing.")

    return plots


def main():
    # ============ 在这里可以单独修改任意晶体的参数 ============
    # 注意：必须在 build_beamline() 之前调用 update_analyzer_config()！

    # 示例：修改第(2,1)号晶体的位置和角度
    print("=" * 60)
    print("正在修改分析晶体参数...")
    print("=" * 60)

    """
    修改第(i,j)号晶体的中心位置坐标（每个分析晶体的初始坐标位置可通过该文件夹的“阵列分析晶体的坐标及姿态.txt”查找，移动相应距离对应的新的中心位置坐标可以通过“新坐标计算.ipynb”计算），适时还需修改pitch和yaw（第i+1列第j+1行）
    """
    update_analyzer_config(0, 0,
                            center=(256.7, 1029.4, 130.3),# 新位置，与默认值有明显差异
                            #pitch=np.radians(81.15),  # 新pitch角，增加0度
                            #yaw=np.radians(-15)# 新yaw角，增加0弧度
                            )#A1
    update_analyzer_config(0, 1, center=(258.216, 1035.259, 0.0))#A2
    update_analyzer_config(0, 2, center=(255.499, 1024.585, -129.690))#A3
    update_analyzer_config(1, 0, center=(129.542, 1055.070, 130.544))#B1
    update_analyzer_config(1, 1, center=(130.3, 1061.0, 0.0))#B2
    update_analyzer_config(1, 2, center=(128.816, 1049.159, -129.812))#B3
    update_analyzer_config(2, 0, center=(0.0, 1063.978, 130.666))#C1
    update_analyzer_config(2, 1, center=(0.0, 1068.9, 0.0))#C2
    update_analyzer_config(2, 2, center=(0.0, 1057.030, -129.812))#C3
    update_analyzer_config(3, 0, center=(-129.542, 1055.070, 130.544))#D1
    update_analyzer_config(3, 1, center=(-130.3, 1061.0, 0.0))#D2
    update_analyzer_config(3, 2, center=(-128.816, 1049.159, -129.812))#D3
    update_analyzer_config(4, 0, center=(-256.940, 1030.363, 130.422))#E1
    update_analyzer_config(4, 1, center=(-258.216, 1035.259, 0.0))#E2
    update_analyzer_config(4, 2, center=(-255.259, 1023.622, -129.569))#E3
    beamLine = build_beamline()


    # 打印分析器信息，特别标记已修改的晶体
    print("=" * 60)
    print("5×3 Analyzer Array Configuration")
    print("=" * 60)
    print("注意: 标记为 [MODIFIED] 的晶体使用了自定义参数")
    print()

    for analyzer in beamLine.analyzers:
        i = analyzer.grid_i
        j = analyzer.grid_j
        modified_mark = "[MODIFIED]" if analyzer.modified else ""
        print(f"Analyzer ({i},{j}) {modified_mark}:")
        print(f"  Name: {analyzer.name}")
        print(f"  Center: [{analyzer.center[0]:.1f}, {analyzer.center[1]:.1f}, {analyzer.center[2]:.1f}] mm")
        print(f"  Pitch: {np.degrees(analyzer.pitch):.4f}°")
        print(f"  Yaw: {np.degrees(analyzer.yaw):.4f}°")

        # 显示默认值作为对比
        default_center, default_pitch, default_yaw = calculate_default_params(i, j)
        if analyzer.modified:
            print(
                f"  [对比默认值] Center: [{default_center[0]:.1f}, {default_center[1]:.1f}, {default_center[2]:.1f}] mm")
            print(f"  [对比默认值] Pitch: {np.degrees(default_pitch):.4f}°")
            print(f"  [对比默认值] Yaw: {np.degrees(default_yaw):.4f}°")
        print()

    # 运行模拟
    plots = define_plots(beamLine)

    print("\n" + "=" * 60)
    print("Starting ray tracing simulation...")
    print("=" * 60)

    xrtr.run_ray_tracing(
        plots, beamLine=beamLine, processes=1 if useTT else nprocesses)

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    print(f"Generated plots:")
    print("  1. 15 detector spot plots (one for each analyzer)")
    print("  2. 15 energy distribution plots on analyzers (one for each analyzer)")
    print("\nGenerated .mat files (for MATLAB processing):")
    print("  detector_0_0.mat ... detector_4_2.mat (15 files)")
    print("\nNote: Check the output for [MODIFIED] markers to confirm custom parameters were used.")


if __name__ == '__main__':
    main()