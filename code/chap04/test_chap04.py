#!/usr/bin/env python3
"""正本から関数定義だけを読み、標準の描画・出力セルから独立に検証します。"""
from pathlib import Path
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
source = (ROOT / "chap04_master.py").read_text(encoding="utf-8")
parsed = ast.parse(source)
functions = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
G = 4.30091e-6 * 1.0227121650537077**2
ns = {"np": np, "G": G}
exec(compile(functions, "chap04_master.py", "exec"), ns)
interp = ns["linear_interp_one"]
trap = ns["cumulative_trapezoid"]
cdf = ns["disk_cdf_exact"]
build = ns["build_radial_cdf"]
make = ns["make_disk"]
accel = ns["acceleration_plummer"]
loop = ns["acceleration_plummer_loop"]
vc = ns["circular_speed_plummer"]
phi = ns["potential_plummer"]
step = ns["leapfrog_step"]
integrate = ns["integrate_leapfrog"]
diag = ns["diagnostics"]
exact = ns["exact_disk_positions"]
metrics = ns["disk_error_metrics"]
empirical = ns["empirical_cdf_distance"]
profile = ns["surface_number_profile"]
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


# 台形則と線形補間は解析的に答えの分かる非一様格子で確認します。
x = np.array([0.0, 0.2, 0.9, 1.5, 3.0])
check("trapezoid: constant", np.allclose(trap(x, np.ones_like(x)), x))
check("trapezoid: affine", np.allclose(trap(x, 2*x+1), x*x+x))
check("trapezoid: shifted grid", np.allclose(trap(x+7, np.ones_like(x)), x))
check("interpolation: interior affine", np.isclose(interp(0.65, x, 2*x+1), 2.3))
check("interpolation: lower endpoint", interp(0.0, x, 2*x+1) == 1.0)
check("interpolation: upper endpoint", interp(3.0, x, 2*x+1) == 7.0)
check("interpolation: exact node", interp(0.9, x, 2*x+1) == 2.8)
check("interpolation: inverse example", interp(2.5, [0,2,3], [0,1,3]) == 2.0)
check("interpolation: arbitrary table matches NumPy", np.allclose([interp(t,x,np.sin(x)) for t in np.linspace(0,3,31)], np.interp(np.linspace(0,3,31),x,np.sin(x))))
raises("interpolation: left extrapolation rejected", ValueError, lambda: interp(-1.0,x,x))
raises("interpolation: right extrapolation rejected", ValueError, lambda: interp(4.0,x,x))
raises("interpolation: NaN query rejected", ValueError, lambda: interp(np.nan,x,x))
raises("interpolation: duplicate nodes rejected", ValueError, lambda: interp(0.0,[0,0,1],[0,1,2]))
raises("interpolation: descending nodes rejected", ValueError, lambda: interp(0.0,[1,0],[1,0]))
raises("interpolation: mismatched shapes rejected", ValueError, lambda: interp(0.0,[0,1],[1]))
raises("interpolation: nonfinite values rejected", ValueError, lambda: interp(0.0,[0,1],[0,np.inf]))
# 線形補間によるx^2の中点誤差はh^2/4です。
errors = [abs(interp(0.5*h, [0,h], [0,h*h]) - (0.5*h)**2) for h in [0.4,0.2,0.1]]
check("interpolation: second-order midpoint errors", np.allclose(np.array(errors[:-1])/errors[1:],4))

mass, a, rd, rmax = 1.0e11, 3.0, 3.0, 15.0
r, f = build(rd,rmax,1024)
check("CDF: endpoints", f[0] == 0.0 and f[-1] == 1.0)
check("CDF: increasing", np.all(np.diff(f)>0))
check("CDF: number of nodes", len(r) == 1025)
check("CDF: nonnegative", np.all((f>=0)&(f<=1)))
check("CDF: analytic match", np.max(abs(f-cdf(r,rd,rmax))) < 2.0e-6)
check("CDF: analytic outside support", np.array_equal(cdf([-1,0,15,20],rd,rmax),[0,0,1,1]))
check("CDF: mass fraction at Rd", abs(cdf(rd,rd,rmax)-(1-2/np.e)/(1-6*np.exp(-5)))<2e-15)
check("CDF: scale invariance", np.allclose(cdf(r,rd,rmax),cdf(2*r,2*rd,2*rmax)))
for count in [8,17,256]:
    rn,fn = build(rd,rmax,count)
    check(f"CDF: inverse endpoints {count}", np.array_equal(np.interp([0,1],fn,rn),[0,rmax]))
raises("CDF: zero rd rejected", ValueError, lambda: build(0,rmax,32))
raises("CDF: negative rmax rejected", ValueError, lambda: build(rd,-1,32))
raises("CDF: zero intervals rejected", ValueError, lambda: build(rd,rmax,0))
raises("CDF: bool intervals rejected", ValueError, lambda: build(rd,rmax,True))

state, radius, angle = make(128,mass,a,rd,rmax,1,np.random.default_rng(4))
repeat,_,_ = make(128,mass,a,rd,rmax,1,np.random.default_rng(4))
reverse,_,_ = make(128,mass,a,rd,rmax,-1,np.random.default_rng(4))
check("disk: shape", state.shape == (128,4))
check("disk: reproducibility", np.array_equal(state,repeat))
check("disk: radial support", np.all((radius>=0)&(radius<=rmax)))
check("disk: angles", np.all((angle>=0)&(angle<2*np.pi)))
check("disk: radial distance", np.allclose(np.linalg.norm(state[:,:2],axis=1),radius))
check("disk: no radial velocity", np.max(abs(np.sum(state[:,:2]*state[:,2:],axis=1))) < 2e-12)
check("disk: circular speeds", np.allclose(np.linalg.norm(state[:,2:],axis=1),vc(radius,mass,a)))
check("disk: spin reverses velocities only", np.array_equal(state[:,:2],reverse[:,:2]) and np.array_equal(state[:,2:],-reverse[:,2:]))
check("disk: positive angular momentum", np.all(state[:,0]*state[:,3]-state[:,1]*state[:,2]>0))
raises("disk: zero count rejected", ValueError, lambda: make(0,mass,a,rd,rmax,1,np.random.default_rng(0)))
raises("disk: invalid spin rejected", ValueError, lambda: make(10,mass,a,rd,rmax,0,np.random.default_rng(0)))
raises("disk: negative mass rejected", ValueError, lambda: make(10,-1,a,rd,rmax,1,np.random.default_rng(0)))
check("acceleration: loop vs vector", np.allclose(accel(state[:,:2],mass,a),loop(state[:,:2],mass,a),rtol=1e-14,atol=1e-12))
check("acceleration: origin", np.array_equal(accel(np.zeros((1,2)),mass,a),np.zeros((1,2))))
check("acceleration: independence from other tracers", np.array_equal(accel(state[:1,:2],mass,a),accel(state[:,:2],mass,a)[:1]))
check("acceleration: sign inversion", np.array_equal(accel(-state[:,:2],mass,a),-accel(state[:,:2],mass,a)))
probe = np.array([[3.,4.]])
hsmall = 1e-4
fd = []
for k in range(2):
    displacement = np.zeros((1,2));displacement[0,k]=hsmall
    fd.append(-(phi(np.linalg.norm(probe+displacement),mass,a)-phi(np.linalg.norm(probe-displacement),mass,a))/(2*hsmall))
check("acceleration: minus potential gradient",np.allclose(accel(probe,mass,a)[0],fd,rtol=1e-8))

# 診断量の軸と規格化を、2粒子の手計算で確かめます。
example = np.array([[[3.,4.,-4.,3.],[0.,2.,1.,0.]]])
rr,ee,ll = diag(example,mass,a)
check("diagnostics: shape",rr.shape==ee.shape==ll.shape==(1,2))
check("diagnostics: radii",np.allclose(rr,[5,2]))
check("diagnostics: angular",np.allclose(ll,[25,-2]))
check("diagnostics: energy",np.allclose(ee,[[12.5+phi(5,mass,a),0.5+phi(2,mass,a)]]))
check("exact disk: t=0",np.allclose(exact([0],state,mass,a,1)[0],state[:,:2]))
check("exact disk: radius invariant",np.allclose(np.linalg.norm(exact([0,.2,.6],state,mass,a,1),axis=2),radius))
zero = np.zeros((1,4))
check("exact disk: central tracer",np.array_equal(exact([0,1],zero,mass,a,1),np.zeros((2,1,2))))
# 円環平均は全Nを分母にし、表示範囲外の粒子を勝手に再規格化しません。
prof,counts,area = profile(np.array([.5,1.5,2.5,8.]),np.array([0.,1.,2.,3.]))
check("profile: annular areas",np.allclose(area,np.pi*np.array([1,3,5])))
check("profile: counts",np.array_equal(counts,[1,1,1]))
check("profile: no renormalization after range cut",np.isclose(np.sum(prof*area),.75))
check("empirical CDF: one quantile",np.isclose(empirical([5.],rd,rmax),max(cdf(5,rd,rmax),1-cdf(5,rd,rmax))))

# KDKの加速度再利用、保存間隔、可逆性を標準図とは独立の小計算で検証します。
calls = [0]
def counted(pos,mass,a):
    calls[0]+=1
    return accel(pos,mass,a)
small = state[:5].copy()
original = small.copy()
t, hist = integrate(counted,small,.025,50,mass,a,7)
check("KDK: force evaluation count",calls[0]==51)
check("KDK: unchanged input",np.array_equal(small,original))
check("KDK: initial history",np.array_equal(hist[0],small))
check("KDK: final time saved",t[-1]==.025)
check("KDK: nondivisible saved count",len(t)==9)
t_full,h_full=integrate(accel,small,.025,50,mass,a,1)
check("KDK: saved subset equals full history",np.array_equal(hist,h_full[np.r_[np.arange(0,51,7),50]]))
t_sparse,h_sparse=integrate(accel,small,.025,50,mass,a,100)
check("KDK: long saving interval keeps endpoints",len(t_sparse)==2 and np.array_equal(h_sparse[-1],h_full[-1]))
last=h_full[-1].copy();last[:,2:]*=-1
_,back=integrate(accel,last,.025,50,mass,a,50)
check("KDK: time reversal position",np.max(abs(back[-1,:,:2]-small[:,:2])) < 1e-11)
check("KDK: time reversal velocity",np.max(abs(back[-1,:,2:]+small[:,2:])) < 1e-10)
check("KDK: finite nondivisible history",np.isfinite(hist).all())
raises("integration: zero steps rejected",ValueError,lambda:integrate(accel,small,.1,0,mass,a))
raises("integration: invalid shape rejected",ValueError,lambda:integrate(accel,[1,2,3,4],.1,10,mass,a))
raises("integration: nonfinite input rejected",ValueError,lambda:integrate(accel,np.full((1,4),np.nan),.1,10,mass,a))
raises("integration: zero save interval rejected",ValueError,lambda:integrate(accel,small,.1,10,mass,a,0))
raises("integration: nonpositive scale rejected",ValueError,lambda:integrate(accel,small,.1,10,mass,0))
raises("integration: NaN end time rejected",ValueError,lambda:integrate(accel,small,np.nan,10,mass,a))
# 別の半径・複数粒子で同時刻の解析円軌道との比較。
custom_pos=np.array([[2.,0.],[0.,4.],[-7.,0.]])
radii=np.linalg.norm(custom_pos,axis=1)
custom_vel=vc(radii,mass,a)[:,None]*np.column_stack((-custom_pos[:,1]/radii,custom_pos[:,0]/radii))
custom=np.column_stack((custom_pos,custom_vel))
conv=[]
for n in [400,800,1600]:
    tt,hh=integrate(accel,custom,.15,n,mass,a,n//100)
    err=metrics(tt,hh,custom,mass,a,rd,1)
    conv.append(err[0])
check("KDK: independent multi-particle second order",3.9<conv[-2]/conv[-1]<4.1)
check("KDK: angular momentum conserved",err[3]<1e-11)
# 同じ1粒子を別の粒子と一緒に解いても、軌道が変わらないことを確認。
_,one=integrate(accel,custom[:1],.15,400,mass,a,4)
_,three=integrate(accel,custom,.15,400,mass,a,4)
check("tracers: independent orbits",np.array_equal(one[:,0],three[:,0]))

(ROOT/"data"/"independent_tests.json").write_text(json.dumps({"passed":len(checks),"failed":0,"checks":checks},ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Independent tests: {len(checks)} passed, 0 failed.")
