export const equationExamples={
  lif:{label:'Adaptive LIF',drive:1.65,spread:.55,dt_ms:.1,duration_ms:600,
    custom:{equations:'dv/dt = (drive - v - w) / (tau * ms) : 1 (unless refractory)\ndw/dt = -w / (tau_w * ms) : 1',parameters:{tau:20,tau_w:120,jump:.06},initial:{v:0,w:0},threshold:'v > 1',reset:'v = 0\nw += jump',refractory_ms:2}},
  adex:{label:'AdEx',drive:18,spread:2,dt_ms:.05,duration_ms:600,
    custom:{equations:'dv/dt = (-(v + 65) + delta * exp((v - vt) / delta) + drive - w) / (tau * ms) : 1 (unless refractory)\ndw/dt = (a * (v + 65) - w) / (tau_w * ms) : 1',parameters:{delta:2,vt:-50,tau:20,tau_w:120,a:.1,jump:2},initial:{v:-65,w:0},threshold:'v > -30',reset:'v = -65\nw += jump',refractory_ms:2}},
  qif:{label:'Quadratic IF',drive:1,spread:.4,dt_ms:.05,duration_ms:600,
    custom:{equations:'dv/dt = (v**2 + drive) / (tau * ms) : 1 (unless refractory)',parameters:{tau:10},initial:{v:-2},threshold:'v > 2',reset:'v = -2',refractory_ms:2}},
  izh:{label:'Izhikevich',drive:10,spread:2,dt_ms:.1,duration_ms:400,
    custom:{equations:'dv/dt = (0.04*v**2 + 5*v + 140 - w + drive) / ms : 1\ndw/dt = a*(b*v - w) / ms : 1',parameters:{a:.02,b:.2,c:-65,d:8},initial:{v:-65,w:-13},threshold:'v >= 30',reset:'v = c\nw += d',refractory_ms:0}}
};
