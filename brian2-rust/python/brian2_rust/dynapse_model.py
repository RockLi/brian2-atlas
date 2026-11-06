"""Explicit Brian2 source model for the narrow DynapSim conversion contract.

Values are dimensionless SI magnitudes (ampere, farad, volt, second), matching
Rockpool's numerical API. This is not an arbitrary LIF-to-analog approximation.
"""
import numpy as np

PARAMETERS=dict(Idc=2.**-40,Igain_ahp=2.**-35,Igain_mem=2.**-35,Igain_syn=2.**-27,
    Ipulse_ahp=2.**-22,Ipulse=2.**-25,Iref=2.**-30,Ispkthr=2.**-24,
    Itau_ahp=2.**-35,Itau_mem=2.**-37,Itau_syn=2.**-34,Iw_ahp=2.**-40,
    C_ahp=2.**-35,C_syn=2.**-35,C_pulse_ahp=2.**-41,C_pulse=2.**-41,
    C_ref=2.**-39,C_mem=2.**-38,Io=2.**-40,kappa_n=.75,kappa_p=.625,Ut=.03125,Vth=.75,Iscale=2.**-27)
STATES=('iahp','imem','isyn','q','timer_ref','vmem')
# Lower bounds use max(x, lower) = lower + (x-lower+abs(x-lower))/2,
# because the source IR clip requires a finite upper bound.
UPDATE="""kk = (kappa_n+kappa_p)/2
tr = Vth*C_ref/Iref
tp = Vth*C_pulse/Ipulse
ta = Vth*C_pulse_ahp/Ipulse_ahp
ts = (Ut/kk)*C_syn/Itau_syn
tahp = (Ut/kk)*C_ahp/Itau_ahp
incoming = stimulus(t,i)*weight*Iscale
isyn_raw = (Igain_syn/Itau_syn)*incoming
isyn_inf = Io+(isyn_raw-Io+abs(isyn_raw-Io))/2
isyn = exp(-.001/ts)*isyn+(1-exp(-tp/ts))*isyn_inf
iahp_raw = exp(-.001/tahp)*iahp+(1-exp(-ta/tahp))*(Igain_ahp/Itau_ahp)*Iw_ahp*q
iahp = Io+(iahp_raw-Io+abs(iahp_raw-Io))/2
feedback = exp((kk**2/(kk+1))*(vmem/Ut))
leak = Itau_mem+iahp
inject_raw = (isyn-leak+Idc)*int(timer_ref==0)
injection = Io+(inject_raw-Io+abs(inject_raw-Io))/2
steady = (Igain_mem/Itau_mem)*(injection-leak)
positive = (Io*feedback/leak)*(imem+Igain_mem)
change = (imem/(((Ut/kk)*C_mem/leak)*(imem+Igain_mem)))*(steady+positive-imem*(1+iahp/Itau_mem))
imem_raw = imem+change*.001
imem = Io+(imem_raw-Io+abs(imem_raw-Io))/2
vmem = (Ut/kk)*log(imem/Io)
count_raw = ceil(log(imem/Ispkthr))
q = (count_raw+abs(count_raw))/2
fired = clip(q,0,1)
imem = (1-fired)*imem+fired*Io
remaining = (timer_ref-.001+abs(timer_ref-.001))/2
timer_ref = (1-fired)*remaining+fired*tr"""


def build(raster,parameters=None):
    import brian2 as b
    b.start_scope();b.set_device('runtime');b.prefs.codegen.target='numpy'
    b.prefs.codegen.loop_invariant_optimisations=False
    raster=np.asarray(raster);n=raster.shape[1]
    values={**PARAMETERS,'weight':1.0,**(parameters or {})}
    table=b.TimedArray(raster,dt=b.ms)
    clock=b.Clock(dt=b.ms,name='dynapse_clock')
    equations='\n'.join([f'{s}:1' for s in STATES]+[f'{p}:1 (constant)' for p in values])
    group=b.NeuronGroup(n,equations,threshold='q>0',clock=clock,name='dynapse',namespace={'stimulus':table})
    for key,value in values.items():setattr(group,key,value)
    for key in ('iahp','imem','isyn'):setattr(group,key,values['Io'])
    group.run_regularly(UPDATE,when='groups',order=1,name='dynapse_update')
    return b.Network(group,b.StateMonitor(group,STATES,record=True,when='end',name='dynapse_state'),
                     b.SpikeMonitor(group,name='dynapse_events'))
