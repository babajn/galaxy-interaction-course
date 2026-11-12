#!/usr/bin/env python3
"""正本から関数定義だけを読み、標準の描画・出力セルから独立に検証します。"""
from pathlib import Path
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
G = 4.30091e-6 * 1.0227121650537077**2
source = (ROOT / "chap05_master.py").read_text(encoding="utf-8")
parsed = ast.parse(source)
functions = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
ns = {"np": np, "G": G}
exec(compile(functions, "chap05_master.py", "exec"), ns)
make = ns["make_initial_state"]
disk = ns["make_disk"]
build = ns["build_radial_cdf"]
trap = ns["cumulative_trapezoid"]
place = ns["place_disk"]
encounter = ns["make_zero_energy_encounter"]
accel = ns["acceleration_system"]
plummer = ns["acceleration_plummer"]
centers = ns["acceleration_centers"]
step = ns["leapfrog_step"]
integ = ns["integrate_leapfrog"]
diag = ns["center_diagnostics"]
errors = ns["center_errors"]
outer = ns["outer_fractions"]
energy = ns["tracer_energy"]
checks = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append({"name": name, "passed": True})


def raises(name, exception, func):
    try:
        func()
    except exception:
        check(name, True)
    else:
        check(name, False)


m = np.array([1e11, 4e10])
a = np.array([3.0, 1.7])
b = 2.4
rd = np.array([3., 1.])
rm = np.array([15., 5.])
counts = np.array([24, 17])
spins = np.array([1, -1])
state, origin, radius = make(m, a, rd, rm, counts, spins, 60., 8., b, 12, 512)
repeat, origin2, radius2 = make(m, a, rd, rm, counts, spins, 60., 8., b, 12, 512)
check("assembly: shape", state.shape == (43, 4))
check("assembly: origin length", origin.shape == (41,))
check("assembly: group sizes", np.array_equal(np.bincount(origin), counts))
check("assembly: fixed order", np.array_equal(origin[:24], np.zeros(24)) and np.all(origin[24:] == 1))
check("assembly: repeatable initial state", np.array_equal(state, repeat))
check("assembly: repeatable labels and radii", np.array_equal(origin, origin2) and np.array_equal(radius, radius2))
check("assembly: center-of-mass position", np.linalg.norm(np.sum(m[:, None]*state[:2, :2],axis=0)/sum(m)) < 1e-14)
check("assembly: center-of-mass velocity", np.linalg.norm(np.sum(m[:, None]*state[:2, 2:],axis=0)/sum(m)) < 1e-14)

local_pos = state[2:, :2] - state[origin, :2]
local_vel = state[2:, 2:] - state[origin, 2:]
check("placement: radii", np.allclose(np.linalg.norm(local_pos, axis=1), radius,rtol=2e-14))
check("placement: circular speed", np.allclose(np.linalg.norm(local_vel,axis=1), ns["circular_speed_plummer"](radius,m[origin],a[origin]),rtol=2e-14))
check("placement: zero radial velocity", np.max(np.abs(np.sum(local_pos*local_vel,axis=1))) < 2e-11)
check("placement: spin signs", np.array_equal(np.sign(local_pos[:,0]*local_vel[:,1]-local_pos[:,1]*local_vel[:,0]),spins[origin]))
check("placement: local bounds", np.all(radius <= rm[origin]))
loc, _, _ = disk(7, m[0],a[0],rd[0],rm[0],1,np.random.default_rng(1))
loccopy = loc.copy()
placed = place(loc,state[0])
check("placement: no input overwrite",np.array_equal(loc,loccopy))
check("placement: inverse transformation",np.allclose(placed-state[0],loc,rtol=2e-14,atol=1e-13))

x=np.array([0.,0.2,0.7,1.3])
check("trapezoid: constant on nonuniform grid",np.allclose(trap(x,np.ones_like(x)),x))
check("trapezoid: affine on nonuniform grid",np.allclose(trap(x,2*x+1),x*x+x))
r,f=build(3.,15.,512)
expected=(1-(1+r/3)*np.exp(-r/3))/(1-6*np.exp(-5))
check("CDF: endpoints",f[0]==0 and f[-1]==1)
check("CDF: strictly increasing",np.all(np.diff(f)>0))
check("CDF: analytic reference",np.max(np.abs(f-expected))<7e-6)
check("CDF: inverse endpoints",np.array_equal(np.interp([0,1],f,r),[0,15]))
raises("CDF: repeated grid rejected",ValueError,lambda:trap([0,0,1],[0,1,2]))
raises("CDF: zero normalization range rejected",ValueError,lambda:build(3,0,10))
raises("CDF: nonpositive count rejected",ValueError,lambda:build(3,15,0))
raises("disk: invalid spin rejected",ValueError,lambda:disk(3,1e11,3,3,15,0,np.random.default_rng(0)))

cstate=encounter(m,60.,8.,b)
d=cstate[1,:2]-cstate[0,:2];u=cstate[1,2:]-cstate[0,2:]
mu=G*sum(m)
specific=.5*np.dot(u,u)-mu/np.sqrt(np.dot(d,d)+b*b)
h=d[0]*u[1]-d[1]*u[0]
check("encounter: incoming",u[0]<0 and u[1]>0)
check("encounter: zero specific energy",abs(specific)/(mu/60)<1e-14)
check("encounter: specified turning point",abs(.5*(h/8)**2-mu/np.sqrt(8**2+b*b))/(mu/8)<1e-14)
c0=encounter(m,60.,8.,0.0);u00=c0[1,2:]-c0[0,2:]
check("encounter: point-mass limit",np.allclose(u00,[-np.sqrt(2*mu/60-2*mu*8/60**2),np.sqrt(2*mu*8)/60],rtol=1e-14))
for ds,dp in [(0.,8.),(8.,8.),(8.,10.),(60.,-1.),(np.nan,8.)]:
    raises(f"encounter: invalid distances {ds},{dp}",ValueError,lambda ds=ds,dp=dp:encounter(m,ds,dp,b))
raises("encounter: negative b",ValueError,lambda:encounter(m,60,8,-1))
raises("model: invalid mass count",ValueError,lambda:ns["check_model"]([1e11],a,b))
raises("model: negative scale",ValueError,lambda:ns["check_model"](m,[3,-1],b))
raises("model: nonfinite mass",ValueError,lambda:ns["check_model"]([np.inf,1],a,b))

A=accel(state[:,:2],m,a,b)
pair=centers(state[:2,:2],m,b)
check("force: output shape",A.shape==(43,2))
check("force: center pair only",np.array_equal(A[:2],pair))
force_scale=G*m[0]*m[1]/60**2
check("force: unequal-mass action reaction",np.linalg.norm(np.sum(m[:,None]*pair,axis=0))/force_scale<1e-15)
check("force: unequal accelerations",np.isclose(np.linalg.norm(pair[0])/np.linalg.norm(pair[1]),m[1]/m[0]))
check("force: center independence of tracers",np.array_equal(accel(state[:2,:2],m,a,b),A[:2]))
subset=state[[0,1,3,9],:2]
check("force: tracer independence of other tracers",np.array_equal(accel(subset,m,a,b)[2:],A[[3,9]]))
check("force: sum of both fields",np.array_equal(A[2:],plummer(state[2:,:2],state[0,:2],m[0],a[0])+plummer(state[2:,:2],state[1,:2],m[1],a[1])))
check("force: center field finite",np.array_equal(plummer(state[0:1,:2],state[0,:2],m[0],a[0]),np.zeros((1,2))))
check("force: arbitrary translation",np.allclose(accel(state[:,:2]+[5,-3],m,a,b),A,rtol=1e-13,atol=1e-11))
check("force: central inversion",np.array_equal(accel(-state[:,:2],m,a,b),-A))
check("force: label exchange",np.allclose(centers(state[:2,:2][::-1],m[::-1],b)[::-1],pair))
check("force: coincident softened centers",np.array_equal(centers(np.zeros((2,2)),m,b),np.zeros((2,2))))
raises("force: coincident unsoftened centers",ValueError,lambda:centers(np.zeros((2,2)),m,0.))
# 有効対エネルギーの中心座標での微分から、力を独立に確認します。
probe=state[:2,:2].copy(); dh=1e-4
for k in range(2):
    shift=np.zeros((2,2));shift[k,0]=dh
    wp=-G*m[0]*m[1]/np.sqrt(np.sum(((probe+shift)[1]-(probe+shift)[0])**2)+b*b)
    wm=-G*m[0]*m[1]/np.sqrt(np.sum(((probe-shift)[1]-(probe-shift)[0])**2)+b*b)
    check(f"force: pair potential derivative body {k}",np.isclose(-(wp-wm)/(2*dh)/m[k],pair[k,0],rtol=1e-8))
# 星のポテンシャル勾配は二つのPlummer場で計算します。
pos_probe=state[[2],:2].copy()
for k in range(2):
    shift=np.zeros((1,2));shift[0,k]=dh
    def phi(pos):
        return sum(-G*m[j]/np.sqrt(np.sum((pos-state[j,:2])**2)+a[j]**2) for j in range(2))
    grad=-(phi(pos_probe+shift)-phi(pos_probe-shift))/(2*dh)
    check(f"force: tracer potential gradient component {k}",np.isclose(grad,A[2,k],rtol=1e-8,atol=1e-7))

# KDKの評価数と保存間隔。小さな系で独立に確認します。
calls=[0]
def counted(pos,masses,scales,b):
    calls[0]+=1
    return accel(pos,masses,scales,b)
t,hist,ct,ch=integ(counted,state,.02,31,m,a,b,10)
check("KDK: force cache count",calls[0]==32)
check("KDK: saved final nonmultiple",len(t)==5 and t[-1]==.02)
check("KDK: center time count",len(ct)==32)
check("KDK: center/full history synchronization",np.array_equal(hist[:, :2],ch[[0,10,20,30,31]]))
check("KDK: input unchanged",np.array_equal(state,repeat))
t2,h2,ct2,ch2=integ(accel,state,.02,31,m,a,b,1)
check("KDK: save interval does not change final state",np.array_equal(hist[-1],h2[-1]))
check("KDK: center history independent of save interval",np.array_equal(ch,ch2))
_,_,_,ch_only=integ(accel,state[:2],.02,31,m,a,b,10)
check("KDK: center trajectory independent of tracers",np.array_equal(ch_only,ch))
p,v=state[:,:2],state[:,2:]
original_a=accel(p,m,a,b)
p1,v1,a1=step(accel,p,v,original_a,1e-4,m,a,b)
p2,v2,_=step(accel,p1,v1,a1,-1e-4,m,a,b)
check("KDK: inverse step",np.allclose(p2,p,rtol=0,atol=1e-13) and np.allclose(v2,v,rtol=0,atol=1e-12))
check("KDK: new acceleration is cached",np.array_equal(a1,accel(p1,m,a,b)))
for nsbad in [0,-1,3.5,True]:
    raises(f"KDK: invalid step count {nsbad}",ValueError,lambda nsbad=nsbad:integ(accel,state,.1,nsbad,m,a,b))
raises("KDK: wrong state shape",ValueError,lambda:integ(accel,np.ones((4,3)),.1,10,m,a,b))
raises("KDK: missing center row",ValueError,lambda:integ(accel,state[:1],.1,10,m,a,b))
raises("KDK: NaN state",ValueError,lambda:integ(accel,state*np.nan,.1,10,m,a,b))
raises("KDK: zero output interval",ValueError,lambda:integ(accel,state,.1,10,m,a,b,0))
raises("KDK: nonfinite final time",ValueError,lambda:integ(accel,state,np.inf,10,m,a,b))

# 小さな手計算で、診断量の軸・質量の重み・二重計上を確認します。
simple=np.array([[[0.,0.,1.,2.],[3.,4.,-2.,1.]]])
sm=np.array([2.,3.]);d=diag(simple,sm,1.)
check("diagnostics: separation",np.array_equal(d['D'],[5.]))
check("diagnostics: momentum",np.array_equal(d['P'],[[-4.,7.]]))
check("diagnostics: center of mass",np.array_equal(d['Rcm'],[[1.8,2.4]]))
check("diagnostics: angular momentum",np.array_equal(d['L'],[33.]))
check("diagnostics: energy one pair",np.allclose(d['E'],[12.5-G*6/np.sqrt(26)]))
zdiag=diag(cstate[None],m,b)
check("diagnostics: zero-energy encounter normalization",np.isfinite(errors(np.array([0.]),cstate[None],m,b)).all())
# 固定した粒子位置でポテンシャルの時間微分を差分評価します。
watch_state=state[[0,1,2]].copy()
ww=watch_state.copy();wm=watch_state.copy();hh=1e-7
ww[:2,:2]+=hh*ww[:2,2:];wm[:2,:2]-=hh*wm[:2,2:]
partial=(energy(ww[None],m,a)[0,0]-energy(wm[None],m,a)[0,0])/(2*hh)
analytic=sum(np.dot(state[j,2:],plummer(state[[2],:2],state[j,:2],m[j],a[j])[0]) for j in range(2))
check("diagnostics: moving-potential energy rate sign",np.isclose(partial,analytic,rtol=2e-8))
example=np.array([[[10,0,0,0],[-10,0,0,0],[11,0,0,0],[-12,0,0,0],[50,0,0,0]]],float)
org=np.array([0,1,1])
check("origin: radii use fixed parent",np.array_equal(ns['origin_radii'](example,org),[[1,2,60]]))
check("origin: outer fraction normalizes each group",np.array_equal(outer(example,org,30),[[0,.5]]))

# 補助検証：相互作用を消した、等速で移動する二つの孤立円盤。
# この試験の星は出身銀河だけの重力を受け、中心は等速直線運動します。
def uncoupled(pos,masses,scales,b):
    out=np.zeros_like(pos)
    for j in range(2):
        rows=2+np.flatnonzero(origin==j)
        out[rows]=plummer(pos[rows],pos[j],masses[j],scales[j])
    return out
iso_end=.03
ti,hi,_,_=integ(uncoupled,state,iso_end,600,m,a,b,600)
exact=np.empty((len(origin),2))
for j in range(2):
    sel=origin==j
    omega=np.sqrt(G*m[j])/(radius[sel]**2+a[j]**2)**.75
    theta=spins[j]*omega*iso_end
    s=local_pos[sel]
    exact[sel,0]=s[:,0]*np.cos(theta)-s[:,1]*np.sin(theta)
    exact[sel,1]=s[:,0]*np.sin(theta)+s[:,1]*np.cos(theta)
    exact[sel]+=state[j,:2]+iso_end*state[j,2:]
check("control: translating isolated circles",np.max(np.linalg.norm(hi[-1,2:,:2]-exact,axis=1))<5e-5)
check("control: translating centers",np.allclose(hi[-1,:2,:2],state[:2,:2]+iso_end*state[:2,2:],atol=1e-12))

# 中心軌道の回転・時間反転を、標準図とは異なる不等質量モデルで確認します。
ts,hs,cts,chs=integ(accel,state,.35,3500,m,a,b,3500)
back=hs[-1].copy();back[:,2:]*=-1
_,hb,_,_=integ(accel,back,.35,3500,m,a,b,3500)
check("interaction: reverse positions",np.max(np.linalg.norm(hb[-1,:,:2]-state[:,:2],axis=1))<1e-7)
check("interaction: reverse velocities",np.max(np.linalg.norm(hb[-1,:,2:]+state[:,2:],axis=1))<1e-5)
cd=diag(chs,m,b)
check("interaction: center momentum",errors(cts,chs,m,b)[2]<1e-12)
check("interaction: center angular momentum",errors(cts,chs,m,b)[1]<1e-12)
check("interaction: center energy accuracy",errors(cts,chs,m,b)[0]<1e-4)
check("interaction: pericenter",abs(np.min(cd['D'])-8)<1e-3)
retro=state.copy(); rr=2+np.flatnonzero(origin==1)
retro[rr,2:]=2*state[1,2:]-retro[rr,2:]
_,hrt,_,crt=integ(accel,retro,.35,3500,m,a,b,3500)
check("retrograde: centers unchanged",np.array_equal(crt,chs))
check("retrograde: other disk unchanged",np.array_equal(hrt[:,2+np.flatnonzero(origin==0)],hs[:,2+np.flatnonzero(origin==0)]))

result={"tests_passed":len(checks),"tests_failed":0,"checks":checks}
(ROOT/'data'/'independent_tests.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(f"Independent tests: {len(checks)} passed, 0 failed.")
