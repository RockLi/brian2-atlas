import {number} from './experiment.js';
export const groupColors={sensory:'#e7a142',local:'#cf7996',projection:'#7a6be1',kenyon:'#36a8a1',lateral_horn:'#548cc4',output:'#a082b9'};
const layouts=new WeakMap();
export function validateCircuit(circuit,template){
  if(circuit.schema!=='b2-flywire-circuit-v1'||circuit.nodes.length!==template.definition.populations[0].count)throw Error('Invalid FlyWire circuit metadata.');
  const syn=template.instance.synapses[0],sensory=template.instance.populations[0].parameters.is_sensory;
  if(!syn||syn.source.length!==circuit.edges.length)throw Error('FlyWire metadata does not match the simulation topology.');
  const roots=new Set();
  circuit.nodes.forEach((node,i)=>{if(!/^\d+$/.test(node.root_id)||BigInt(node.root_id)>18446744073709551615n||roots.has(node.root_id))throw Error('Invalid FlyWire root ID.');roots.add(node.root_id);if(number(sensory[i])!==Number(node.group==='sensory'))throw Error('Sensory cell annotation mismatch.');});
  circuit.edges.forEach((edge,k)=>{if(syn.source[k]!==edge.source||syn.target[k]!==edge.target||number(syn.parameters.signed_contacts[k])!==edge.signed_contacts)throw Error('FlyWire edge or weight mismatch.');});
  return circuit;
}
function layout(canvas,circuit){
  const width=Math.max(700,canvas.getBoundingClientRect().width),height=canvas.getBoundingClientRect().height;
  const old=layouts.get(canvas);if(old?.width===width&&old?.height===height&&old.circuit===circuit)return old;
  const positions=[],groups=circuit.groups,span=(width-40)/groups.length;
  groups.forEach((group,k)=>{
    const all=circuit.nodes.flatMap((node,i)=>node.group===group.id?[i]:[]),ids=all.length<=96?all:Array.from({length:96},(_,j)=>all[Math.round(j*(all.length-1)/95)]),columns=Math.min(6,Math.ceil(Math.sqrt(ids.length))),rows=Math.ceil(ids.length/columns);
    ids.forEach((id,j)=>{positions[id]={x:20+span*(k+.5)+(j%columns-(columns-1)/2)*Math.min(12,span/(columns+1)),y:188+(Math.floor(j/columns)-(rows-1)/2)*Math.min(16,240/Math.max(1,rows-1))};});
  });
  const result={width,height,positions,circuit,edges:circuit.edges.filter(e=>positions[e.source]&&positions[e.target]).sort((a,b)=>b.contacts-a.contacts).slice(0,600)};layouts.set(canvas,result);return result;
}
export function renderNetwork(canvas,data,{cutoff,probe,selected}){
  const circuit=data.network;if(!circuit)return;
  const l=layout(canvas,circuit),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(l.width*dpr);canvas.height=Math.round(l.height*dpr);
  const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#fff';c.fillRect(0,0,l.width,l.height);
  const active=new Set();let lo=0,hi=data.spikes;while(lo<hi){const mid=(lo+hi)>>>1;if(data.ticks[mid]*data.dtMs<cutoff-10)lo=mid+1;else hi=mid;}for(let k=lo;k<data.spikes;k++){const time=data.ticks[k]*data.dtMs;if(time>cutoff)break;if(time>=cutoff-10)active.add(data.indices[k]);}
  const span=(l.width-40)/circuit.groups.length;
  circuit.groups.forEach((group,k)=>{c.fillStyle=groupColors[group.id]+'09';c.fillRect(12+span*k,56,span-4,270);c.fillStyle=groupColors[group.id];c.textAlign='center';c.font='600 10px system-ui';c.fillText(group.label,20+span*(k+.5),24);c.font='9px ui-monospace,monospace';c.fillText(`${group.count} cells`,20+span*(k+.5),41);});
  for(const edge of l.edges){const from=l.positions[edge.source],to=l.positions[edge.target],cut=data.config.sensory_output===0&&circuit.nodes[edge.source].group==='sensory',hot=active.has(edge.source)&&!cut&&edge.signed_contacts!==0;
    c.strokeStyle=cut?'#c9ced41a':edge.signed_contacts===0?'#9ca4b520':edge.signed_contacts<0?(hot?'#c36587aa':'#c3658720'):(hot?'#7971d599':'#7971d51a');
    c.lineWidth=hot?1.3:.55;c.beginPath();c.moveTo(from.x,from.y);c.bezierCurveTo(from.x+(to.x-from.x)*.4,from.y-12,to.x-(to.x-from.x)*.4,to.y+12,to.x,to.y);c.stroke();if(hot){const angle=Math.atan2(-12,(to.x-from.x)*.4);c.beginPath();c.moveTo(to.x-4*Math.cos(angle-.5),to.y-4*Math.sin(angle-.5));c.lineTo(to.x,to.y);c.lineTo(to.x-4*Math.cos(angle+.5),to.y-4*Math.sin(angle+.5));c.stroke();}
  }
  const stimulating=data.config.sensory_weight_mv>0&&data.config.sensory_rate_hz>0&&cutoff>=data.config.stimulus_start_ms&&cutoff<data.config.stimulus_end_ms;
  circuit.nodes.forEach((node,i)=>{if(!l.positions[i])return;const {x,y}=l.positions[i],hot=active.has(i);if(hot||stimulating&&node.group==='sensory'){c.fillStyle=hot?'#e8b55d3a':'#e8b55d15';c.beginPath();c.arc(x,y,hot?7:5,0,2*Math.PI);c.fill();}c.fillStyle=hot?'#f4b54c':groupColors[node.group];c.beginPath();c.arc(x,y,hot?3.6:2.4,0,2*Math.PI);c.fill();if(i===probe||i===selected){c.strokeStyle=i===selected?'#263048':'#17a898';c.lineWidth=1.4;c.beginPath();c.arc(x,y,5.5,0,2*Math.PI);c.stroke();}});
  c.fillStyle='#8c94a6';c.textAlign='left';c.font='9px system-ui';c.fillText(`${l.positions.filter(Boolean).length} cells / ${l.edges.length} edges shown · All ${circuit.nodes.length.toLocaleString()} cells / ${circuit.edges.length.toLocaleString()} edges simulated`,20,l.height-12);
}
export function networkCellAt(canvas,circuit,event){
  const l=layout(canvas,circuit),box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;
  let best=-1,distance=9;for(let i=0;i<l.positions.length;i++){const p=l.positions[i];if(!p)continue;const d=Math.hypot(x-p.x,y-p.y);if(d<distance){best=i;distance=d;}}return best;
}
export function groupRates(data){
  return data.network.groups.map(group=>{const ids=data.network.nodes.flatMap((node,i)=>node.group===group.id?[i]:[]);const spikes=ids.reduce((sum,i)=>sum+data.counts[i],0);return {...group,spikes,hz:spikes/ids.length/(data.durationMs/1000)};});
}
