#!/usr/bin/env python3
"""第7章の動力学的摩擦・RK4・衛星テスト粒子の独立検証。"""
from pathlib import Path
from math import erf
import ast, json, numpy as np
ROOT=Path(__file__).resolve().parent
source=(ROOT/'chap07_master.py').read_text(encoding='utf-8')
module=ast.parse(source)
definitions=ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef)],type_ignores=[])
G=4.30091e-6*1.0227121650537077**2
ns=dict(np=np,erf=erf,G=G,Path=Path)
exec(compile(definitions,str(ROOT/'chap07_master.py'),'exec'),ns)
results=[]
def check(name,condition): results.append({'name':name,'passed':bool(condition)})
def close(name,a,b,rtol=1e-11,atol=1e-12): check(name,np.allclose(a,b,rtol=rtol,atol=atol))
def rejects(name,function,exception=ValueError):
    try: function()
    except exception: check(name,True)
    else: check(name,False)

p=dict(m_host=1e11,a_host=3.,m_sat=1e10,a_sat=1.5,ln_lambda=float(np.log(11)),friction=True)
no=dict(p,friction=False)
x=np.array([20.0,1.0]); v=np.array([40.,100.])
a=ns['chandrasekhar_acceleration'](x,v,p)
check('drag removes satellite orbital energy',a@v<0)
close('drag is parallel to minus velocity',a[0]*v[1]-a[1]*v[0],0,atol=1e-10)
close('drag reverses with velocity',ns['chandrasekhar_acceleration'](x,-v,p),-a)
close('zero-speed limit finite and zero',ns['chandrasekhar_acceleration'](x,np.zeros(2),p),[0,0],atol=0)
close('zero Coulomb log gives zero drag',ns['chandrasekhar_acceleration'](x,v,dict(p,ln_lambda=0)),[0,0],atol=0)
close('fixed-background drag proportional to satellite mass',ns['chandrasekhar_acceleration'](x,v,dict(p,m_sat=2e10)),2*a)
close('drag proportional to fixed Coulomb log',ns['chandrasekhar_acceleration'](x,v,dict(p,ln_lambda=2*p['ln_lambda'])),2*a)
slow=np.array([1e-6,0.0])
close('low speed drag linear',ns['chandrasekhar_acceleration'](x,2*slow,p),2*ns['chandrasekhar_acceleration'](x,slow,p),rtol=1e-12,atol=1e-20)
# 独立な積分表現 B(X)/X^3=4/sqrt(pi)*int_0^1 u^2 exp[-X^2u^2]du。
# これは検証だけで使うGauss求積で、教材の実装依存を増やしません。
z,w=np.polynomial.legendre.leggauss(48); z=(z+1)/2; w=w/2
for xx in [0.0,1e-12,1e-4,1e-3,.1,1.,3.,5.]:
    reference=4/np.sqrt(np.pi)*np.sum(w*z*z*np.exp(-xx*xx*z*z))
    close(f'Maxwell factor independent quadrature X={xx:g}',ns['maxwell_factor_over_x3'](xx),reference,rtol=3e-9,atol=1e-13)
rejects('negative X rejected',lambda:ns['maxwell_factor_over_x3'](-1.0))
rejects('nonfinite X rejected',lambda:ns['maxwell_factor_over_x3'](np.nan))
rejects('negative logLambda rejected',lambda:ns['check_df_params'](dict(p,ln_lambda=-1)))
rejects('boolean logLambda rejected',lambda:ns['check_df_params'](dict(p,ln_lambda=True)))
rejects('negative satellite mass rejected',lambda:ns['check_df_params'](dict(p,m_sat=-1)))
rejects('friction flag requires boolean',lambda:ns['check_df_params'](dict(p,friction='yes')))
rejects('center RHS wrong row count rejected',lambda:ns['df_center_rhs'](0,np.zeros((2,4)),p))

# 背景密度と与えたsigmaが等方的平衡の微分関係を満たすか。
for r in [1.,3.,20.]:
    dr=1e-4
    pressure=lambda r: ns['host_density'](r,p['m_host'],p['a_host'])*ns['host_sigma'](r,p['m_host'],p['a_host'])**2
    derivative=(pressure(r+dr)-pressure(r-dr))/(2*dr)
    reference=-ns['host_density'](r,p['m_host'],p['a_host'])*G*p['m_host']*r/(r*r+p['a_host']**2)**1.5
    close(f'Plummer pressure derivative r={r:g}',derivative,reference,rtol=2e-8,atol=0)

vc=ns['circular_speed_plummer'](20,p['m_host'],p['a_host'])
s0=np.array([[20.,0.,0.,vc]])
original=s0.copy()
count_rhs=[0]
def counted_rhs(t,state,params):
    count_rhs[0]+=1
    return ns['df_center_rhs'](t,state,params)
t,h=ns['integrate_rk4'](counted_rhs,s0,.05,50,p,7)
check('RK4 n steps uses 4n RHS calls',count_rhs[0]==200)
close('RK4 leaves initial array unchanged',s0,original,atol=0,rtol=0)
close('RK4 always stores final state',t[-1],.05)
_,hh=ns['integrate_rk4'](ns['df_center_rhs'],s0,.05,50,dict(p,ln_lambda=0))
_,hn=ns['integrate_rk4'](ns['df_center_rhs'],s0,.05,50,no)
close('friction=False equals logLambda=0',hh,hn,atol=0,rtol=0)
rejects('zero step count rejected',lambda:ns['integrate_rk4'](ns['df_center_rhs'],s0,1,0,p))
rejects('nonfinite initial state rejected',lambda:ns['integrate_rk4'](ns['df_center_rhs'],np.full((1,4),np.nan),1,20,p))

# 摩擦ありなしの長い標準軌道を、図とは独立に再計算。
t,hd=ns['integrate_rk4'](ns['df_center_rhs'],s0,1.5,1500,p)
_,hn=ns['integrate_rk4'](ns['df_center_rhs'],s0,1.5,1500,no)
rd,ed,ld,power=ns['df_diagnostics'](hd,p)
rn,en,ln,_=ns['df_diagnostics'](hn,no)
close('no-drag circle radius',rn,20.0,atol=1e-7,rtol=0)
check('no-drag energy conserved within tolerance',np.max(np.abs(en-en[0]))/abs(en[0])<1e-9)
check('drag energy monotonically decreases in standard run',np.all(np.diff(ed)<0))
check('drag angular momentum decreases in standard run',np.all(np.diff(ld)<0))
check('standard friction final radius in expected range',7.79<rd[-1]<7.81)
check('every stored drag power nonpositive',np.all(power<=0))
work=ns['cumulative_trapezoid'](t,power)
check('energy change agrees with postprocessed drag work',np.max(np.abs(ed-ed[0]-work))/abs(ed[0])<2e-5)

local,_,_=ns['make_disk'](10,1e10,1.5,1.,4.,1,np.random.default_rng(2),64)
full=np.vstack((s0,ns['place_disk'](local,s0[0])))
d_yes=ns['df_tracer_rhs'](0,full,p)
d_no=ns['df_tracer_rhs'](0,full,no)
close('friction does not directly act on star RHS',d_yes[1:],d_no[1:],atol=0,rtol=0)
close('center RHS includes drag only once',d_yes[0,2:]-d_no[0,2:],ns['chandrasekhar_acceleration'](s0[0,:2],s0[0,2:],p))
_,ht=ns['integrate_rk4'](ns['df_tracer_rhs'],full,.05,100,p,10)
_,hs=ns['integrate_rk4'](ns['df_center_rhs'],s0,.05,100,p,10)
close('RK4 center independent of tracer population',ht[:,:1],hs,atol=0,rtol=0)

passed=sum(r['passed'] for r in results)
(ROOT/'data').mkdir(exist_ok=True)
(ROOT/'data/independent_tests.json').write_text(json.dumps({'passed':passed,'failed':len(results)-passed,'checks':results},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Independent tests: {passed} passed, {len(results)-passed} failed.')
for r in results:
    if not r['passed']: print('FAILED:',r['name'])
if passed!=len(results): raise SystemExit(1)
