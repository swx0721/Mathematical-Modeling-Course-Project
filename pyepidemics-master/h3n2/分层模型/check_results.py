import pandas as pd, os
base = r'd:\HP\OneDrive\Desktop\学校\课程\专业课\数学建模\课程项目\Mathematical-Modeling-Course-Project\pyepidemics-master\h3n2\分层模型'
for sc, cn in [('normal','常态'),('sporadic','散发'),('cluster','聚集')]:
    sp = os.path.join(base, cn)
    sm = pd.read_csv(os.path.join(sp, f'summary_{sc}.csv'))
    row = sm.iloc[0]
    ts = pd.read_csv(os.path.join(sp, f'timeseries_{sc}.csv'))
    N0 = ts['N'].iloc[0]
    ar = ts['R'].iloc[-1] / N0
    peak_I = ts['I'].max()
    peak_day = ts.loc[ts['I'].idxmax(), 'day']
    print(f'{cn}: J={row["J"]:.4f}, AR={ar:.4f}, peak_I={peak_I:.1f}, peak_day={peak_day:.0f}')
