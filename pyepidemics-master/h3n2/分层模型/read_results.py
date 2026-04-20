import json
from pathlib import Path

base = Path(r'd:\HP\OneDrive\Desktop\学校\课程\专业课\数学建模\课程项目\Mathematical-Modeling-Course-Project\pyepidemics-master\h3n2\分层模型')

for sc, folder, sn in [('常态', '常态', 'normal'), ('散发', '散发', 'sporadic'), ('聚集', '聚集', 'cluster')]:
    p = base / folder
    jf = list(p.glob('opt_result_' + sn + '.json'))
    if jf:
        with open(jf[0], encoding='utf-8-sig') as f:
            jr = json.load(f)
        print('=== ' + sc + ' ===')
        print('  J=' + str(jr.get('objective_J')) + ', A=' + str(jr.get('epidemic_loss_A')) + ', C=' + str(jr.get('control_cost_C')) + ', D=' + str(jr.get('disruption_D')))
        print('  attack_rate=' + str(jr.get('attack_rate')) + ', peak_I=' + str(jr.get('peak_I')) + ', peak_I_day=' + str(jr.get('peak_I_day')))
        ctrls = jr.get('controls', {})
        for k, v in ctrls.items():
            print('    ' + k + ': ' + str(round(v, 4)))
    else:
        print(sc + ': json not found')
    print()
