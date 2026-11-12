#!/usr/bin/env python3
"""本章の新規処理と再利用した物理の独立検証。標準の図を作らずに実行できます。"""
from pathlib import Path
from math import erf
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'chap06_master.py').read_text(encoding='utf-8')
module = ast.parse(source)
definitions = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
G = 4.30091e-6 * 1.0227121650537077**2
ns = dict(np=np, erf=erf, G=G, Path=Path, t_end=0.60, threshold=30.0, tau_target=0.20)
exec(compile(definitions, str(ROOT/'chap06_master.py'), 'exec'), ns)
results = []


def check(name, condition):
    results.append({'name': name, 'passed': bool(condition)})


def close(name, a, b, rtol=1e-11, atol=1e-12):
    check(name, np.allclose(a, b, rtol=rtol, atol=atol))


def rejects(name, function, exception=ValueError):
    try:
        function()
    except exception:
        check(name, True)
    else:
        check(name, False)


ap, ac = ns['acceleration_plummer'], ns['acceleration_centers']
ms, ss = np.array([1e11, 3e10]), np.array([3.0, 3.0])
pos = np.array([[-6.0, 2.0], [9.0, -4.0]])
acc = ac(pos, ms, 3.0)
close('unequal-mass action reaction', ms @ acc, [0.0, 0.0], atol=0.01)
d = pos[1]-pos[0]
close('center relative acceleration', acc[1]-acc[0], -G*ms.sum()*d/(d@d+9)**1.5)
close('translated Plummer field', ap(pos+5, np.ones(2)*5, 1e11, 3.0), ap(pos,np.zeros(2),1e11,3.0))
close('Plummer acceleration at its center',ap(np.zeros((1,2)),np.zeros(2),1e11,3.0),np.zeros((1,2)))
for q in [1.0, 0.5, 0.1]:
    mass=np.array([1e11,q*1e11])
    ini=ns['make_zero_energy_encounter'](mass,60.0,8.0,3.0)
    dr,dv=ini[1,:2]-ini[0,:2],ini[1,2:]-ini[0,2:]
    energy=0.5*np.dot(dv,dv)-G*mass.sum()/np.sqrt(dr@dr+9.0)
    close(f'zero energy q={q}',energy,0.0,atol=1e-10)
    close(f'zero barycenter q={q}',mass@ini[:,:2]/mass.sum(),[0,0],atol=2e-14)
    check(f'incoming radial speed q={q}',np.dot(dr,dv)<0)

# 近点合わせのテストには、既知の最小を持つ合成履歴を使います。
ct=np.arange(5)*0.1
ch=np.zeros((5,2,4)); ch[:,1,0]=[4,2,1,2,4]
j,tp,dm,tau=ns['aligned_index'](ct,ct,ch,0.15)
close('known sampled pericenter', [tp,dm], [.2,1])
check('nearest stored post-pericenter epoch',j in (3,4) and abs(tau-.15)<=.05000000001)
rejects('aligned epoch outside range', lambda:ns['aligned_index'](ct,ct,ch,1.0))
end_ch=ch.copy(); end_ch[:,1,0]=[1,2,3,4,5]
rejects('pericenter at edge rejected',lambda:ns['aligned_index'](ct,ct,end_ch,0.1))

h=np.zeros((1,6,4)); h[0,1,0]=100
h[0,2:,0]=[29,31,70,69]
origin=np.array([0,0,1,1])
close('origin distances, not nearest center',ns['origin_radii'](h,origin),[[29,31,30,31]])
close('strict threshold and separate denominators',ns['outer_fractions'](h,origin,30),[[.5,.5]])

cfg=dict(masses=np.array([1e11,1e11]),scales=np.array([3.,3.]),rds=np.array([3.,3.]),rmaxs=np.array([15.,15.]),counts=np.array([16,16]),spins=np.array([1,1]),d_start=60.,d_peri=8.,b=3.,seed=42,n_cdf=128)
ini,ori,_=ns['make_initial_state'](**cfg)
ini2,ori2,_=ns['make_initial_state'](**dict(cfg,spins=np.array([1,-1])))
close('paired seeds keep all initial positions',ini[:,:2],ini2[:,:2],atol=0,rtol=0)
rows=2+np.flatnonzero(ori==1)
close('only internal velocity is reversed',ini2[rows,2:],2*ini[1,2:]-ini[rows,2:])
close('unchanged center initial state',ini[:2],ini2[:2],atol=0,rtol=0)
call_count=[0]
def counted_acc(p,m,s,b):
    call_count[0]+=1
    return ns['acceleration_system'](p,m,s,b)
tt,hh,tc,hc=ns['integrate_leapfrog'](counted_acc,ini,.02,40,cfg['masses'],cfg['scales'],3.,7)
check('KDK caches acceleration: n+1 calls',call_count[0]==41)
close('last time saved even off stride',tt[-1],.02)
_,_,_,hc_only=ns['integrate_leapfrog'](ns['acceleration_system'],ini[:2],.02,40,cfg['masses'],cfg['scales'],3.,7)
close('tracers do not change center orbit',hc,hc_only,atol=0,rtol=0)


# 中心間有効長さの解析式だけを独立に確認します。
for D in [5.0, 8.0, 12.0]:
    ratio=D**3/(D**2+3.0**2)**1.5
    check(f'softening ratio bounded D={D:g}',0.0<ratio<1.0)
check('softening ratio approaches one far away', abs(1e6**3/(1e12+9.0)**1.5-1.0)<1e-10)

passed=sum(r['passed'] for r in results)
(ROOT/'data').mkdir(exist_ok=True)
(ROOT/'data/independent_tests.json').write_text(json.dumps({'passed':passed,'failed':len(results)-passed,'checks':results},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Independent tests: {passed} passed, {len(results)-passed} failed.')
for r in results:
    if not r['passed']: print('FAILED:',r['name'])
if passed!=len(results): raise SystemExit(1)
