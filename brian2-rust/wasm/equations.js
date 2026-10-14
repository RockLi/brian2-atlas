// Small, explicit expression language. Never evaluates JavaScript or Python.
// Brian2 checks the resulting typed, dimensioned AtlasIR again inside WASM.
import {bits} from './experiment.js';
const dim=[0,0,0,0,0,0,0],time=[0,0,1,0,0,0,0];
const literal=x=>({op:'literal',bits:bits(x)}),load=name=>({op:'load',name}),binary=(op,left,right)=>({op,left,right});
const namePattern=/^[a-zA-Z][a-zA-Z0-9_]{0,31}$/;
const reserved=new Set(['dt','t','i','N','ms','second','exp','exprel','and','or','not','not_refractory','lastspike','spike']);
export function expression(source,allowed){
  if(typeof source!=='string'||source.length>2048)throw Error('Expressions must contain at most 2,048 characters.');
  const tokens=[];let pos=0;
  while(pos<source.length){const m=/^(\s+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\*\*|>=|<=|==|!=|[()+\-*/<>])/.exec(source.slice(pos));if(!m)throw Error(`Unexpected expression character at column ${pos+1}.`);pos+=m[0].length;if(!/^\s/.test(m[0]))tokens.push(m[0]);}
  if(tokens.length>256)throw Error('Expression exceeds 256 tokens.');
  let at=0,depth=0;const precedence={or:1,and:2,'==':3,'!=':3,'>':3,'>=':3,'<':3,'<=':3,'+':4,'-':4,'*':5,'/':5,'**':7};
  const ops={or:'or',and:'and','==':'eq','!=':'ne','>':'gt','>=':'ge','<':'lt','<=':'le','+':'add','-':'sub','*':'mul','/':'div','**':'pow'};
  function parse(min=0){
    if(++depth>32)throw Error('Expression nesting exceeds 32 levels.');
    const token=tokens[at++];let left;
    if(token==='-'||token==='+'||token==='not'){const arg=parse(token==='not'?3:6);left=token==='+'?arg:{op:token==='-'?'neg':'not',arg};}
    else if(token==='('){left=parse();if(tokens[at++]!==')')throw Error('Expected a closing parenthesis.');}
    else if(token&&/^(\d|\.)/.test(token)){const value=Number(token);if(!Number.isFinite(value))throw Error('Numbers must be finite.');left=literal(value);}
    else if(token==='exp'||token==='exprel'){if(tokens[at++]!=='(')throw Error(`Use ${token}(expression).`);const arg=parse();if(tokens[at++]!==')')throw Error('Expected a closing parenthesis.');left={op:token,arg};}
    else if(allowed.has(token))left=load(token);
    else throw Error(`Unknown symbol: ${token??'end of expression'}.`);
    while(precedence[tokens[at]]>=min){const op=tokens[at++],right=parse(precedence[op]+(op==='**'?0:1));if(op==='**'&&(right.op!=='literal'||!Number.isInteger(fromBits(right.bits))||fromBits(right.bits)>8||fromBits(right.bits)<0))throw Error('Powers require a literal integer exponent from 0 to 8.');left=binary(ops[op],left,right);}
    depth--;return left;
  }
  const result=parse();
  const cost=e=>{const value=e.op==='pow'?cost(e.left)*Math.max(1,fromBits(e.right.bits)):1+['left','right','arg'].reduce((sum,k)=>sum+(e[k]?cost(e[k]):0),0);if(value>512)throw Error('Expression expansion exceeds the 512-operation limit.');return value;};cost(result);
  if(at!==tokens.length)throw Error(`Unexpected token: ${tokens[at]}.`);return result;
}
function fromBits(bits){const b=new DataView(new ArrayBuffer(8));b.setBigUint64(0,BigInt('0x'+bits));return b.getFloat64(0);}
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
export function parseEquations(spec){
  if(!object(spec))throw Error('Custom model must be an object.');
  const keys=['equations','parameters','initial','threshold','reset','refractory_ms'];
  if(Object.keys(spec).some(k=>!keys.includes(k))||keys.some(k=>!Object.hasOwn(spec,k)))throw Error('Custom model requires equations, parameters, initial, threshold, reset and refractory_ms.');
  if(typeof spec.equations!=='string'||spec.equations.length>8192||typeof spec.reset!=='string'||spec.reset.length>4096)throw Error('Equation or reset text exceeds the editor limit.');
  if(!object(spec.parameters)||!object(spec.initial)||Object.keys(spec.parameters).length>24)throw Error('Use JSON objects for initial values and up to 24 parameters.');
  if(!Number.isFinite(spec.refractory_ms)||spec.refractory_ms<0||spec.refractory_ms>20)throw Error('Refractory period must be between 0 and 20 ms.');
  const lines=spec.equations.split('\n').map(s=>s.split('#')[0].trim()).filter(Boolean),states=[];
  for(const line of lines){const match=/^d([A-Za-z][A-Za-z0-9_]*)\/dt\s*=\s*(.*?)\s*:\s*1\s*(\(unless refractory\))?$/.exec(line);if(!match)throw Error(`Expected dvariable/dt = expression : 1 (optional: unless refractory). Received: ${line}`);const [,name,rhs,frozen]=match;if(!namePattern.test(name)||reserved.has(name)||states.some(s=>s.name===name))throw Error(`Invalid or duplicate state: ${name}.`);states.push({name,rhs,frozen:!!frozen});}
  if(states.length<1||states.length>4||!states.some(s=>s.name==='v'))throw Error('Define 1–4 state variables, including v for the main trace.');
  for(const [name,value] of Object.entries(spec.parameters)){if(!namePattern.test(name)||reserved.has(name)||['drive'].includes(name)||states.some(s=>s.name===name))throw Error(`Invalid or reserved parameter: ${name}.`);if(typeof value!=='number'||!Number.isFinite(value)||Math.abs(value)>1e12)throw Error(`Parameter ${name} must be a finite number within ±1e12.`);}
  const names=states.map(s=>s.name);
  if(names.includes('drive')||names.some(s=>!Object.hasOwn(spec.initial,s))||Object.keys(spec.initial).some(s=>!names.includes(s)))throw Error('Provide one numeric initial value for every state. drive is reserved for the input control.');
  for(const value of Object.values(spec.initial))if(typeof value!=='number'||!Number.isFinite(value)||Math.abs(value)>1e12)throw Error('Initial values must be finite numbers within ±1e12.');
  const allowed=new Set([...names,...Object.keys(spec.parameters),'drive','ms','second','t','dt']);
  for(const s of states)s.ast=expression(s.rhs,allowed);
  const threshold=expression(spec.threshold,allowed),resets=[];
  for(const line of spec.reset.split(/[;\n]/).map(s=>s.split('#')[0].trim()).filter(Boolean)){const m=/^([A-Za-z][A-Za-z0-9_]*)\s*(\+=|-=|=)\s*(.+)$/.exec(line);if(!m||!names.includes(m[1]))throw Error('Reset statements must assign to a state using =, += or -=.');if(resets.length>=16)throw Error('At most 16 reset assignments are supported.');const rhs=expression(m[3],allowed);resets.push({name:m[1],ast:m[2]==='='?rhs:binary(m[2]==='+='?'add':'sub',load(m[1]),rhs)});}
  return {states,threshold,resets};
}
export function makeEquationDraft(template,c){
  const spec=c.custom,{states,threshold,resets}=parseEquations(spec),m=structuredClone(template),p=m.definition.populations[0],instance=m.instance.populations[0];
  if(Math.abs(spec.refractory_ms/c.dt_ms-Math.round(spec.refractory_ms/c.dt_ms))>1e-7)throw Error('Custom refractory_ms must be a multiple of dt_ms.');
  const n=c.neurons,steps=Math.round(c.duration_ms/c.dt_ms),record=Array.from({length:Math.min(12,n)},(_,k)=>Math.round(k*(n-1)/(Math.min(12,n)-1)));
  const variable=(name,index_domain='neuron',dimensions=dim)=>({name,dtype:'f64',dimensions:[...dimensions],index_domain});
  p.states=states.map(s=>variable(s.name)).sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0);
  p.parameters=[variable('drive'),variable('ms','scalar',time),variable('second','scalar',time),...Object.keys(spec.parameters).map(k=>variable(k,'scalar'))].sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0);
  p.count=n;p.steps=steps;p.dt=bits(c.dt_ms/1000);p.refractory={mode:'fixed',frozen_states:states.filter(s=>s.frozen).map(s=>s.name).sort()};
  p.monitor={variables:['v',...states.map(s=>s.name).filter(s=>s!=='v')],record,window_steps:steps};p.state_monitors[0]={...p.state_monitors[0],variables:p.monitor.variables,output_variables:[...p.monitor.variables],record};
  const assignment=(target,value,condition=null,dtype='f64')=>({target,dtype,dimensions:[...dim],value,condition});
  // All derivatives use the old state (simultaneous Euler), then commit together.
  const updates=states.map((s,k)=>assignment('_next'+k,binary('add',load(s.name),binary('mul',load('dt'),s.ast))));
  updates.push(...states.map((s,k)=>assignment(s.name,load('_next'+k),s.frozen?'not_refractory':null)));
  p.code_objects[0].scalar=[];p.code_objects[0].vector=updates;
  p.code_objects[1].vector=[assignment('_cond',threshold,null,'bool')];p.code_objects[1].scalar=[];
  p.code_objects[2].vector=resets.map(s=>assignment(s.name,s.ast));p.code_objects[2].scalar=[];
  const globalNames=new Set([...p.states,...p.parameters].map(x=>x.name).concat(['dt','t','not_refractory']));
  function reads(e,set){if(e.op==='load'&&globalNames.has(e.name))set.add(e.name);for(const key of ['left','right','arg'])if(e[key])reads(e[key],set);}
  for(const code of p.code_objects){const read=new Set(),write=new Set();for(const a of code.vector){reads(a.value,read);if(a.condition)read.add(a.condition);if(states.some(s=>s.name===a.target))write.add(a.target);}code.effects={reads:[...read].sort(),writes:[...write].sort()};}
  const resource=name=>states.some(s=>s.name===name)?'population/0/state/'+name:p.parameters.some(s=>s.name===name)?'population/0/parameter/'+name:name==='not_refractory'?'population/0/refractory/'+name:null;
  const lastWriter=new Map(),readers=new Map();
  for(const node of m.definition.schedule.nodes){let read=[],write=[];
    if(node.operation==='state_monitor')read=p.monitor.variables.map(resource);
    else if(node.operation==='spike_monitor')read=['population/0/event/spike'];
    else {const code=p.code_objects[node.item_index];read=code.effects.reads.map(resource).filter(Boolean);write=code.effects.writes.map(resource).filter(Boolean);if(code.kind==='threshold')write.push('population/0/event/spike');if(code.kind==='reset')read.push('population/0/event/spike');}
    node.effects={reads:[...new Set(read)].sort(),writes:[...new Set(write)].sort()};const dependencies=new Set();
    for(const r of [...read,...write])if(lastWriter.has(r))dependencies.add(lastWriter.get(r));
    for(const r of write){for(const reader of readers.get(r)??[])dependencies.add(reader);lastWriter.set(r,node.id);readers.set(r,new Set());}
    for(const r of read){if(!readers.has(r))readers.set(r,new Set());readers.get(r).add(node.id);}node.dependencies=[...dependencies].sort();
  }
  instance.initial_state=Object.fromEntries(states.map(s=>[s.name,Array(n).fill(bits(spec.initial[s.name]))]));
  let seed=c.seed>>>0;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  instance.parameters={...Object.fromEntries(Object.entries(spec.parameters).map(([k,v])=>[k,[bits(v)]])),drive:Array.from({length:n},()=>bits(c.drive+(c.synchronize?0:(random()-.5)*c.spread))),ms:[bits(.001)],second:[bits(1)]};
  instance.refractory={period:bits(spec.refractory_ms/1000),period_ticks:Math.round(spec.refractory_ms/c.dt_ms),initial_lastspike:Array(n).fill(bits(-10000)),initial_not_refractory:Array(n).fill(true)};
  m.definition.clocks[0].dt=p.dt;m.instance.neuron_count=n;m.instance.rng_seed=c.seed;m.run.duration=bits(c.duration_ms/1000);m.run.clocks[0].steps=steps;
  return m;
}
