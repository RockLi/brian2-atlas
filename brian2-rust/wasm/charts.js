import { firingRate } from './experiment.js';
import {presentationFor} from './models.js';
const ink='#9198ac',grid='#eef0f6',purple='#7570df',teal='#28a99f';
function frame(canvas,xmax,ymax,xlabel,ylabel,ymin=0) {
  const box=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.max(100,box.width),h=box.height;
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#fff';c.fillRect(0,0,w,h);
  const p={l:49,r:w-20,t:20,b:h-32};
  const x=v=>p.l+v/(xmax>0?xmax:1)*(p.r-p.l),y=v=>p.b-(v-ymin)/(ymax>ymin?ymax-ymin:1)*(p.b-p.t);
  c.font='9px ui-monospace, SFMono-Regular, monospace';c.lineWidth=1;
  for(let k=0;k<=4;k++) {const yy=p.t+(p.b-p.t)*k/4;c.strokeStyle=grid;c.beginPath();c.moveTo(p.l,yy);c.lineTo(p.r,yy);c.stroke();c.fillStyle=ink;c.textAlign='right';c.fillText((ymin+(ymax-ymin)*(1-k/4))[ymax-ymin<.1?'toExponential':'toFixed'](ymax-ymin<3?1:0),p.l-9,yy+3);}
  for(let k=0;k<=6;k++){const xx=x(xmax*k/6);c.textAlign='center';c.fillStyle=ink;c.fillText((xmax*k/6)[xmax<1?'toPrecision':'toFixed'](xmax<1?2:xmax<10?1:0),xx,p.b+16);}
  c.textAlign='left';c.fillStyle=ink;c.font='8px system-ui';c.fillText(ylabel,p.l,10);c.textAlign='right';c.fillText(xlabel,p.r,h-3);
  return {c,p,x,y,w,h};
}
function path(f,points,color,width=1.5,fill=false) {
  if(!points.length)return;const {c}=f;c.strokeStyle=color;c.lineWidth=width;c.beginPath();points.forEach(([x,y],k)=>k?c.lineTo(f.x(x),f.y(y)):c.moveTo(f.x(x),f.y(y)));c.stroke();
  if(fill){c.lineTo(f.x(points.at(-1)[0]),f.y(0));c.lineTo(f.x(points[0][0]),f.y(0));c.closePath();c.fillStyle='#7871e312';c.fill();}
}
export function renderCharts(data,{cutoff,binMs,probe,previous}) {
  const $=s=>document.getElementById(s);
  const r=frame($('raster'),data.durationMs,data.n-1,'time / ms','neuron #');
  r.c.save();r.c.beginPath();r.c.rect(r.p.l,r.p.t,r.p.r-r.p.l,r.p.b-r.p.t);r.c.clip();
  if(data.network){r.c.fillStyle='#eabf571b';r.c.fillRect(r.x(data.config.stimulus_start_ms),r.p.t,r.x(data.config.stimulus_end_ms)-r.x(data.config.stimulus_start_ms),r.p.b-r.p.t);}
  for(let j=0,stride=Math.max(1,Math.ceil(data.spikes/60000));j<data.spikes;j+=stride){const t=data.ticks[j]*data.dtMs;if(t>cutoff)break;const neuron=data.indices[j];r.c.fillStyle=neuron===probe?'#21aa9b':`hsla(${238+neuron/data.n*12},55%,${55+neuron/data.n*10}%,.68)`;r.c.fillRect(r.x(t),r.y(neuron)-1,1.35,2.3);}
  if(cutoff<data.durationMs){r.c.strokeStyle='#22aa9a';r.c.beginPath();r.c.moveTo(r.x(cutoff),r.p.t);r.c.lineTo(r.x(cutoff),r.p.b);r.c.stroke();}r.c.restore();
  const rates=firingRate(data,binMs),oldRates=previous&&previous.durationMs===data.durationMs?firingRate(previous,binMs):[],maxRate=Math.max(5,...rates,...oldRates)*1.16;
  const rate=frame($('rate'),data.durationMs,maxRate,'time / ms','Hz / neuron');
  if(oldRates.length){path(rate,Array.from(oldRates,(v,k)=>[Math.min((k+.5)*binMs,data.durationMs),v]),'#c7cad6',1);}
  path(rate,Array.from(rates,(v,k)=>[Math.min((k+.5)*binMs,data.durationMs),v]).filter(p=>p[0]<=cutoff),purple,1.65,true);
  const hz=Array.from(data.counts,k=>k/(data.durationMs/1000)),maxHz=hz.reduce((a,b)=>Math.max(a,b),5)*1.05,bins=16,hist=Array(bins).fill(0);
  for(const v of hz)hist[Math.min(bins-1,Math.floor(v/maxHz*bins))]++;
  const distribution=frame($('distribution'),maxHz,Math.max(...hist)*1.16||1,'rate / Hz','neurons');
  hist.forEach((v,k)=>{const x=distribution.x(k/bins*maxHz),w=distribution.x((k+1)/bins*maxHz)-x;distribution.c.fillStyle=k%2?'#9793e7':'#817be0';distribution.c.fillRect(x+2,distribution.y(v),Math.max(1,w-4),distribution.p.b-distribution.y(v));});
  const meta=presentationFor(data),lane=Math.max(0,data.record.indexOf(probe)),stride=data.record.length;
  const primary=meta.primary??'v';if(!data.record.length||!data.trace[primary])return;
  // Keep each bucket's extrema, in time order, so narrow spikes survive downsampling.
  const pointsFor=name=>{
    const values=data.trace[name],points=[],limit=Math.min(data.steps,Math.floor(cutoff/data.dtMs)+1),bucket=Math.max(1,Math.ceil(limit/900));
    for(let start=0;start<limit;start+=bucket){let low=start,high=start;const end=Math.min(limit,start+bucket);
      for(let j=start;j<end;j++){if(values[j*stride+lane]<values[low*stride+lane])low=j;if(values[j*stride+lane]>values[high*stride+lane])high=j;}
      for(const j of [...new Set([start,low,high,end-1])].sort((a,b)=>a-b))points.push([j*data.dtMs,values[j*stride+lane]]);
    }return points;
  };
  const voltage=pointsFor(primary),aux=meta.aux.map(pointsFor);
  const extent=(series,min,max)=>{for(const points of series)for(const [,v]of points){min=Math.min(min,v);max=Math.max(max,v);}const pad=Math.max(data.presentation?1e-12:.05,(max-min)*.08);return [min-pad,max+pad];};
  const baseline=meta.voltageRange??(data.model==='flywire'?[-70,-40]:meta.threshold===1?[0,1.2]:[-80,40]);const [vmin,vmax]=extent([voltage],...baseline);
  const trace=frame($('trace'),data.durationMs,vmax,'time / ms',meta.voltageUnit,vmin);if(data.network){trace.c.fillStyle='#eabf571b';trace.c.fillRect(trace.x(data.config.stimulus_start_ms),trace.p.t,trace.x(data.config.stimulus_end_ms)-trace.x(data.config.stimulus_start_ms),trace.p.b-trace.p.t);}if(meta.threshold!==null){trace.c.strokeStyle='#c7cada';trace.c.setLineDash([4,4]);trace.c.beginPath();trace.c.moveTo(trace.p.l,trace.y(meta.threshold));trace.c.lineTo(trace.p.r,trace.y(meta.threshold));trace.c.stroke();trace.c.setLineDash([]);}
  path(trace,voltage,purple,1.4);
  const [amin,amax]=data.model==='hodgkin_huxley'?[0,1]:extent(aux,0,meta.aux[0]==='w'?.2:1);
  const auxiliary=frame($('aux-trace'),data.durationMs,amax,'time / ms',meta.auxUnit,amin);
  aux.forEach((points,k)=>path(auxiliary,points,[teal,'#dc9b45','#ce6c92'][k],1.4));

}
export function emptyCharts(){for(const [id,x,y,label]of[['raster',600,160,'neuron #'],['rate',600,50,'Hz / neuron'],['distribution',50,160,'neurons'],['trace',600,1.2,'voltage'],['aux-trace',600,1,'recovery / channel gates']]){const f=frame(document.getElementById(id),x,y,'',label);f.c.fillStyle='#bcc0cd';f.c.font='11px system-ui';f.c.textAlign='center';f.c.fillText('Run an experiment to see results',f.w/2,f.h/2);}}
