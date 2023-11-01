import pandas as pd
import numpy as np
import math

df = pd.read_csv("./ofr_2.csv")

# 设置常数, 所有数据单位都是基本单位
mu = 4.0e-3
rho = 1050
k = 1
L_OCT = 0.18e-3  # OCT切片相隔距离, 0.18mm

area = df[['Area']].to_numpy()/1000000

An = area.max()
As = area.min()

F = 8*math.pi*mu*L_OCT*np.sum(An/area/area)
S = rho/2*(An/As-1)**2
F_mmHg_s_div_cm = F*0.0075*0.01
S_mmHg_s2_div_cm2 = S*0.0075*0.01*0.01
print("F = %f Pa s/m"%F)
print("F = {:.3} mmHg s/cm".format(F_mmHg_s_div_cm))
print("S = %f kg/m^3"%S)
print("S = {:.3} mmHg s^2/cm^s".format(S_mmHg_s2_div_cm2))

F = F_mmHg_s_div_cm
S = S_mmHg_s2_div_cm2

SFR_max_dia = 4.2  # 舒张 diastolic
SFR_max_sys = 2    # 收缩 systolic

b = F+4.5/SFR_max_dia
SFR_dia = (math.sqrt(b**2+360*S)-b)/40/S
print("舒张期SFR={:.3}".format(SFR_dia))

b = F+4.5/SFR_max_sys
SFR_sys = (math.sqrt(b**2+360*S)-b)/40/S
print("收缩期SFR={:.3}".format(SFR_sys))

V_dia = 15
V_sys = 10

Delta_P_dia = F*V_dia*SFR_dia + S*(V_dia*SFR_dia)**2
Delta_P_sys = F*V_sys*SFR_sys + S*(V_sys*SFR_sys)**2

print("舒张期Delta P={:.3} mmHg".format(Delta_P_dia))
print("收缩期Delta P={:.3} mmHg".format(Delta_P_sys))

P_dia = 60
P_sys = 120
P_mean = 80

FFR = (2/3*(P_dia - Delta_P_dia) + 1/3*(P_sys - Delta_P_sys))/P_mean
print("FFR={:.3} mmHg".format(FFR))

