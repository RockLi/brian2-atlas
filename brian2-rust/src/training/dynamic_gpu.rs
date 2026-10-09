//! v5 GPU action ABI. Ordered execution and VJP run on device; Rust commits
//! the optimizer and checks the complete host/device allocation budget.
use super::*;

#[cfg(not(unix))]
pub(super) fn execute(_: &Plan, _: State, _: &[Vec<Vec<f64>>], _: &[usize], _: &[Vec<f64>],
    _: usize, _: &str, _: Option<&mpi::Context>, _: u64, _: u64, _: bool, _: &itinerary::Tape,_:Option<&poisson_cache::Checkpoint>) -> Result<Output> {
    Err("dynamic GPU training requires a Unix host".into())
}

#[cfg(unix)]
pub(super) fn execute(plan: &Plan, mut state: State, inputs: &[Vec<Vec<f64>>], labels: &[usize],
    initial: &[Vec<f64>], bytes: usize, operation: &str, mpi: Option<&mpi::Context>,
    _start_tick: u64, sequence: u64, default_initial: bool, itinerary: &itinerary::Tape,poisson_state:Option<&poisson_cache::Checkpoint>) -> Result<Output> {
    ensure(inputs.iter().flatten().flatten().all(|&x| x == 0.0 || x == 1.0), "dynamic event inputs must be binary spikes")?;
    let spec = plan.dynamic.as_ref().unwrap();
    let (_, n) = plan.validate()?;
    let b = inputs.len(); let frames = inputs[0].len(); let t = itinerary.len(); let i = plan.sizes[0];
    let width = spec.initial.len(); let classes = *plan.sizes.last().unwrap();
    let edges = state.weights.iter().map(Vec::len).sum::<usize>();
    let reads = spec.actions.iter().map(|a| a.reads.len()).sum::<usize>();
    let address_words = spec.actions.iter().filter_map(|a| a.indirect.as_ref()).map(|v| {
        v.reads.values().map(|r| 2 + r.tables.len()*2 + r.tables.iter().map(Vec::len).sum::<usize>()).sum::<usize>()
        + v.writes.values().map(|r| 3 + r.tables.len()*2 + r.tables.iter().map(Vec::len).sum::<usize>()).sum::<usize>()
    }).sum::<usize>() + spec.actions.iter().filter(|a| a.indirect.is_some()).map(|a| 2+a.reads.len()+a.writes.len()).sum::<usize>();
    let ordinary_tape_width = reads + spec.actions.iter().filter(|a| a.indirect.is_some()).map(|a| a.reads.len()+2*a.writes.len()).sum::<usize>();
    let writes = spec.actions.iter().map(|a| a.writes.len()).sum::<usize>();
    let nodes = spec.program_sets.iter().flatten().map(Vec::len).sum::<usize>();
    let programs = spec.program_sets.iter().map(Vec::len).sum::<usize>();
    let poisson_sets: Vec<bool> = spec.program_sets.iter().map(|s|s.iter().flatten().any(|n|matches!(n,equation::Node::Poisson{..}))).collect();
    let action_poisson = |a: &dynamic::Action| a.program_set.is_some_and(|i|poisson_sets[i]);
    let boundary = operation != "evaluate" && poisson_ir::boundary_possible(spec);
    let poisson_actions = spec.actions.iter().filter(|a|action_poisson(a)).count();
    let ordinary_width = ordinary_tape_width + if boundary {poisson_actions} else {0};
    let shared=poisson_sets.iter().any(|&used|used)||poisson_state.is_some();
    let incoming=poisson_state.map_or(0,|s|s.entries.len());
    let cache=if shared {Some(poisson_gpu_cache::Layout::build(plan,itinerary,ordinary_width,b,poisson_state)?)}else{None};
    let tape_width=cache.as_ref().map_or(ordinary_width,|c|c.width);
    let delta_width=if shared {194}else{130};
    let noise_size = |a: &dynamic::Action| if shared && action_poisson(a) {113}
        else if boundary && action_poisson(a) {97}
        else if action_poisson(a) {16+4*a.noise_streams} else {a.noise_streams};
    let noise_width = spec.actions.iter().map(noise_size).sum::<usize>();
    let clocks = itinerary.width;
    // u128 admission arithmetic precedes all staging/tape allocations. This
    // includes the simultaneous Rust and device copies and per-sample scratch.
    let map_entries = spec.parameter_maps.iter().map(Vec::len).sum::<usize>();
    let timed_nodes = spec.program_sets.iter().flatten().flatten().filter(|node| matches!(node, equation::Node::TimedParameter { .. })).count();
    let elapsed_nodes = spec.program_sets.iter().flatten().flatten().filter(|node| matches!(node, equation::Node::ElapsedCompare { .. })).count();
    let precise_time = elapsed_nodes != 0 || spec.program_sets.iter().flatten().flatten().any(|node| matches!(node, equation::Node::TimeWord { .. }));
    let meta_count = 38u128 + 3*u128::from(shared) + if boundary {2} else {0} + spec.spike_buffers.len() as u128 + address_words as u128 + 4 * (timed_nodes + elapsed_nodes) as u128 + if precise_time { (t*clocks) as u128 } else { 0 } + 1 + (t*(1+clocks)+spec.actions.len()) as u128 + 16 * spec.actions.len() as u128 + (reads + writes + 4 * nodes + 3 * programs + 4 * width + 2 * n + map_entries) as u128;
    let param_count = (3 + n + edges + nodes) as u128 + t as u128 * (1 + clocks) as u128
        + b as u128 * t as u128 * noise_width as u128 +3*incoming as u128;
    let floats = b as u128 * t as u128 * (i + tape_width + n) as u128 + edges as u128
        + b as u128 * (3 * width + edges + classes + 1) as u128;
    let gpu_bytes = 2 * (meta_count * 8 + param_count * 4 + floats * 4 + b as u128 * 8)
        + b as u128 * n as u128 * 4 + b as u128 * 16384
        + t as u128 * clocks as u128 * 8
        + spec.parameter_maps.len() as u128 * 8
        + if mpi.is_some() { b as u128 * delta_width * 4 + delta_width * 12 } else { 0 }
        + if boundary {b as u128 * t as u128 * poisson_actions as u128 * 8} else {0}
        + cache.as_ref().map_or(0,|c|c.host_bytes as u128);
    ensure(gpu_bytes + bytes as u128 <= plan.max_tape_bytes as u128, "native dynamic GPU tape budget exceeded")?;
    let gpu_bytes = gpu_bytes as usize;
    ensure(!shared || (param_count<=u32::MAX as u128 && b as u128*t as u128*tape_width as u128<=u32::MAX as u128),"GPU Poisson cache uint32 address budget exceeded")?;
    fn f32_value(x: f64) -> Result<f32> {
        let y = x as f32; ensure(y.is_finite(), "dynamic GPU value cannot be represented in float32")?; Ok(y)
    }
    fn cast(values: impl Iterator<Item=f64>, count: usize) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(count);
        for value in values { out.push(f32_value(value)?); }
        ensure(out.len() == count, "dynamic GPU staging shape mismatch")?;
        Ok(out)
    }
    let mut parameter_offsets = Vec::with_capacity(state.weights.len()+1); parameter_offsets.push(0usize);
    for row in &state.weights { parameter_offsets.push(parameter_offsets.last().unwrap() + row.len()); }
    // Reserve admitted sizes once: geometric Vec growth could otherwise make
    // the large noise/input staging capacity exceed the element-count budget.
    let mut meta = Vec::with_capacity(meta_count as usize); meta.resize(if shared {36}else{33},0u64);if shared {meta[33]=2;}
    let mut params = Vec::with_capacity(param_count as usize);
    params.extend(cast([plan.surrogate.slope, plan.surrogate.scale, plan.logit_scale].into_iter(),3)?);
    meta[..13].copy_from_slice(&[b as u64,t as u64,i as u64,n as u64,width as u64,edges as u64,classes as u64,
        spec.actions.len() as u64,tape_width as u64,(operation != "evaluate") as u64,plan.tbptt_window.unwrap_or(0) as u64,0,0]);
    meta[13] = meta.len() as u64; meta.resize(meta.len() + 16 * spec.actions.len(), 0);
    let mut map_offsets = Vec::with_capacity(spec.parameter_maps.len());
    for row in &spec.parameter_maps { map_offsets.push(meta.len() as u64); meta.extend(row.iter().map(|&i| i as u64)); }
    let mut headers = Vec::with_capacity(spec.program_sets.len());
    for (set_index,set) in spec.program_sets.iter().enumerate() {
        let h = meta.len(); headers.push(h); meta.resize(h + 3 * set.len(), 0);
        for (s, program) in set.iter().enumerate() {
            let descriptors = meta.len();
            for node in program {
                if let equation::Node::TimedParameter { bank, rows, columns, k, epsilon, .. } = *node {
                    ensure(f32_value(epsilon)? > 0.0, "TimedArray time resolution underflows float32")?;
                    meta.extend_from_slice(&[parameter_offsets[bank] as u64, rows as u64, columns as u64, k]);
                } else if let equation::Node::ElapsedCompare { right, kind, constant, .. } = *node {
                    meta.extend_from_slice(&[right as u64,kind as u64,constant.is_some() as u64,constant.unwrap_or(0.0).to_bits()]);
                }
            }
            let mut timed_offset = descriptors;
            meta[h+3*s] = program.len() as u64 | if program.iter().any(|n|matches!(n,equation::Node::Poisson{..})) {256} else {0} | if boundary && poisson_sets[set_index] {512} else {0} | if shared && poisson_sets[set_index] {1024}else{0}; meta[h+3*s+1] = meta.len() as u64; meta[h+3*s+2] = params.len() as u64;
            for node in program {
                let (mut row, value) = equation::gpu_node(node, &parameter_offsets);
                if let equation::Node::Poisson{rate,..}=*node {row[3]=poisson_ir::rate_differentiable(program,rate) as u64;}
                if row[0] == 24 || row[0] == 37 { row[1] = map_offsets[row[1] as usize]; }
                if row[0] == 25 || row[0] == 51 { row[3] = timed_offset as u64; timed_offset += 4; }
                meta.extend_from_slice(&row); params.push(f32_value(value)?);
            }
        }
    }
    fn append_tables(meta: &mut Vec<u64>, tables: &[Vec<usize>]) {
        let at=meta.len(); meta.resize(at+tables.len()*2,0);
        for (d,row) in tables.iter().enumerate() {
            meta[at+2*d]=row.len() as u64; meta[at+2*d+1]=meta.len() as u64;
            meta.extend(row.iter().map(|&k|k as u64));
        }
    }
    let mut context = 0usize; let mut noise_offset = 0usize;
    for (a, action) in spec.actions.iter().enumerate() {
        let h = meta[13] as usize + 16 * a;
        meta[h] = context as u64; context += action.reads.len();
        meta[h+1] = action.reads.len() as u64; meta[h+2] = meta.len() as u64;
        meta.extend(action.reads.iter().map(|&k| k as u64));
        meta[h+3] = action.writes.len() as u64; meta[h+4] = meta.len() as u64;
        meta.extend(action.writes.iter().map(|&k| k as u64));
        meta[h+5] = action.program_set.map_or(0, |p| headers[p]) as u64;
        meta[h+6] = action.threshold.map_or(0, |k| k+1) as u64;
        if let Some(gate) = &action.trigger {
            meta[h+7] = if gate.state {3} else if gate.external {1} else {2};
            meta[h+8] = if gate.state { action.reads.iter().position(|&k| k == gate.index).unwrap() } else {gate.index} as u64;
        }
        meta[h+9] = action.detach_trigger as u64;
        meta[h+10] = action.mask.map_or(1, |[bank,index]| plan.masks[bank][index] as u64);
        meta[h+11] = action.parameter_index as u64;
        meta[h+12] = noise_offset as u64; noise_offset += noise_size(action);
        meta[h+13] = action.owner as u64;
        meta[h+14] = if action.threshold_predicate { 3 } else if action.threshold_margin { if action.threshold_inclusive { 2 } else { 1 } } else { 0 };
        if let Some(access)=&action.indirect {
            let desc=meta.len();meta[h+15]=desc as u64;meta.resize(desc+2,0);
            let read_map=meta.len();meta[desc]=read_map as u64;meta.resize(read_map+action.reads.len(),0);
            let write_map=meta.len();meta[desc+1]=write_map as u64;meta.resize(write_map+action.writes.len(),0);
            for (&slot,r) in &access.reads {
                meta[read_map+slot]=meta.len() as u64;
                meta.extend_from_slice(&[r.index as u64,r.tables.len() as u64]);append_tables(&mut meta,&r.tables);
            }
            for (&slot,r) in &access.writes {
                meta[write_map+slot]=meta.len() as u64;
                let (kind,index)=match r.index {indirect::Index::Read{slot}=>(0,slot),indirect::Index::Output{slot}=>(1,slot)};
                meta.extend_from_slice(&[kind,index as u64,r.tables.len() as u64]);append_tables(&mut meta,&r.tables);
            }
            context+=action.reads.len()+2*action.writes.len();
        }
        if boundary && action_poisson(action) {context+=1;}
    }
    meta[15] = meta.len() as u64; meta.extend(spec.voltage.iter().map(|&k| k as u64));
    meta[16] = meta.len() as u64; meta.extend(spec.detached.iter().map(|&v| v as u64));
    meta[17] = meta.len() as u64; meta.resize(meta.len()+width,0);
    for &k in &spec.binary_states { let at = meta[17] as usize+k; meta[at] = 1; }
    meta[18] = meta.len() as u64; meta[19] = params.len() as u64;
    for l in 0..plan.sizes.len()-1 { for j in 0..plan.sizes[l+1] {
        meta.push(plan.threshold_reference(l,j).map_or(0, |[bank,k]| 1+parameter_offsets[bank]+k) as u64);
        params.push(f32_value(plan.threshold[l])?);
    }}
    meta[20] = params.len() as u64; params.extend(cast(plan.masks.iter().flatten().copied(),edges)?);
    meta[21] = params.len() as u64;
    params.extend(cast((0..t).map(|k| itinerary.row(k)[0]),t)?);
    meta[22] = params.len() as u64; meta[23] = clocks as u64;
    if precise_time {
        meta[29]=meta.len() as u64;
        meta.extend(itinerary.times.iter().map(|x|x.to_bits()));
    }
    params.extend(cast(itinerary.times.iter().copied(),t*clocks)?);
    meta[24] = params.len() as u64; meta[25] = noise_width as u64;
    let uniform_masks: Vec<u16> = spec.program_sets.iter().map(|p| equation::noise_masks(p.iter()).1).collect();
    for sample in 0..b { for tick in 0..t { for action in &spec.actions {
        for stream in 0..action.noise_streams {
            let uniform = uniform_masks[action.program_set.unwrap()] & (1 << stream) != 0;
            params.push(noise::device_sample(noise::event_sample(action.event_noise.as_ref(),uniform,plan.seed,sequence,sample as u64,action.noise_domain,
                action.noise_entity,itinerary.tick(tick,action),stream as u64), uniform));
        }
        if action_poisson(action) {
            let mut keys=[0.;80];
            poisson_ir::encode_keys(&mut keys,plan,action,sequence,sample,itinerary.tick(tick,action));
            params.extend(std::iter::repeat(0.).take(16-action.noise_streams));
            let key_end=16+4*action.noise_streams;
            params.extend(keys[16..key_end].iter().map(|&x|x as f32));
            if boundary||shared {params.resize(params.len()+97-key_end,0.0);}
            if let Some(cache)=&cache {for stream in 0..16 {params.push(f32::from_bits(cache.offset(action,stream,tick,sample,itinerary)?));}}
        }
    }}}
    meta[26] = default_initial as u64; meta[27] = meta.len() as u64;
    meta.extend(spec.initial_parameters.iter().map(|r| r.map_or(0, |[bank,k]| 1+parameter_offsets[bank]+k) as u64));
    meta[28] = meta.len() as u64; meta.resize(meta.len()+width,0);
    for &k in &spec.integer_states { let at=meta[28] as usize+k; meta[at]=1; }
    meta[30] = meta.len() as u64; meta.push(frames as u64);
    for visit in 0..t {
        meta.push(itinerary.frames[visit].map_or(0,|f|f+1) as u64);
        meta.extend(itinerary.active[visit*clocks..(visit+1)*clocks].iter().map(|&x|u64::from(x)));
    }
    meta[31] = meta.len() as u64;meta.extend(spec.actions.iter().map(|a|a.clock.unwrap_or(0) as u64));
    if !spec.spike_buffers.is_empty() {meta[32]=meta.len() as u64;meta.extend(spec.spike_buffers.iter().map(|&k|k as u64));}
    meta[14] = meta.len() as u64;
    meta.extend_from_slice(&[mpi.map_or(1,|m|m.size) as u64,mpi.map_or(0,|m|m.rank) as u64,0,0,0]);
    if boundary {meta.extend_from_slice(&[0,0]);} // forced noise offset+1, stream
    if let (Some(cache),Some(state))=(&cache,poisson_state){
        meta[34]=params.len() as u64;meta[35]=incoming as u64;
        for entry in &state.entries {
            params.extend_from_slice(&[f32::from_bits(cache.identity_offset(entry.identity,t)?),f32::from_bits(entry.count as u32),entry.rate as f32]);
        }
    }
    meta[11] = meta.len() as u64; meta[12] = params.len() as u64;
    ensure(meta.len() as u128 == meta_count && params.len() as u128 == param_count, "dynamic GPU metadata admission mismatch")?;
    let mut x=Vec::with_capacity(b*t*i);
    for sample in inputs {for frame in &itinerary.frames {
        if let Some(frame)=frame {for &value in &sample[*frame] {x.push(f32_value(value)?);}}
        else {x.resize(x.len()+i,0.0);}
    }}
    let integer_parameters: std::collections::HashSet<_> = spec.integer_parameters.iter()
        .map(|&[bank,k]| parameter_offsets[bank]+k).collect();
    let pack = |value:f64, integer:bool| -> Result<f32> {
        if integer { Ok(f32::from_bits(equation::int32(value)? as u32)) } else { f32_value(value) }
    };
    let mut w=Vec::with_capacity(edges);
    for (k,&x) in state.weights.iter().flatten().enumerate() { w.push(pack(x,integer_parameters.contains(&k))?); }
    let mut v=Vec::with_capacity(b*width);
    for (k,&x) in initial.iter().flatten().enumerate() { v.push(pack(x,meta[meta[28] as usize+k%width]!=0)?); }
    ensure(w.len()==edges && v.len()==b*width, "dynamic GPU staging shape mismatch")?;
    let y = labels.iter().map(|&k| k as u64).collect::<Vec<_>>();
    let mut tape = vec![0f32;b*t*tape_width]; let mut spikes = vec![0f32;b*t*n];
    let mut live = vec![0f32;b*width]; let mut grad = vec![0f32;b*edges]; let mut adj = vec![0f32;b*width];
    let mut logits = vec![0f32;b*classes]; let mut losses = vec![0f32;b]; let mut error = vec![0 as c_char;2048];
    use std::ffi::{c_char,c_void,CStr,CString};
    #[cfg_attr(target_os="macos",link(name="System"))]
    #[cfg_attr(not(target_os="macos"),link(name="dl"))]
    extern "C" { fn dlopen(p:*const c_char,flags:i32)->*mut c_void; fn dlsym(h:*mut c_void,s:*const c_char)->*mut c_void; fn dlclose(h:*mut c_void)->i32; }
    struct Library(*mut c_void); impl Drop for Library {fn drop(&mut self){unsafe{dlclose(self.0);}}}
    let path = CString::new(std::env::var(if plan.backend=="cuda" {"B2_TRAIN_CUDA_LIB"} else {"B2_TRAIN_METAL_LIB"})
        .map_err(|_| "dynamic GPU requires the selected native library; refusing CPU fallback")?)?;
    let handle = unsafe{dlopen(path.as_ptr(),2)}; ensure(!handle.is_null(),"cannot load dynamic GPU library")?;
    let _library = Library(handle);
    // Appended math opcodes require this capability even when legacy layout
    // symbols match; an old shader must never silently treat them as zero.
    let math_symbol=CString::new("b2_train_math_v1")?;
    let math_function=unsafe{dlsym(handle,math_symbol.as_ptr())};
    ensure(!math_function.is_null(),"GPU math ABI capability missing")?;
    let math_version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(math_function)};
    ensure(unsafe{math_version()}==1,"GPU math ABI capability mismatch")?;

    if spec.program_sets.iter().flatten().any(equation::bitwise_used) {
        let name=CString::new("b2_train_bitwise_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU bitwise capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU bitwise capability mismatch")?;
    }

    if spec.program_sets.iter().flatten().any(equation::sequence_used) {
        let name=CString::new("b2_train_sequence_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU sequence capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU sequence capability mismatch")?;
    }
    if spec.program_sets.iter().flatten().any(equation::eager_boolean_used) {
        let name=CString::new("b2_train_boolean_eager_v1")?;let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU eager Boolean capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU eager Boolean capability mismatch")?;
    }
    if shared {
        let name=CString::new("b2_train_poisson_shared_v1")?;let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU shared Poisson cache capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU shared Poisson cache capability mismatch")?;
        let name=CString::new("b2_train_poisson_persistent_v1")?;let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU persistent Poisson cache capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU persistent Poisson cache capability mismatch")?;
    }
    if poisson_sets.iter().any(|&used|used) {
        let name=CString::new("b2_train_poisson_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU Poisson ABI capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU Poisson ABI capability mismatch")?;
    }
    if operation != "evaluate" && poisson_ir::boundary_possible(spec) {
        let name=CString::new("b2_train_poisson_vjp_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU Poisson rate VJP capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU Poisson rate VJP capability mismatch")?;
    }
    if boundary {
        let name=CString::new("b2_train_poisson_boundary_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU Poisson boundary replay capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU Poisson boundary replay capability mismatch")?;
    }
    if operation != "evaluate" {
        let name=CString::new("b2_train_vjp_activity_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU VJP activity capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU VJP activity capability mismatch")?;
    }
    let symbol = CString::new(if plan.backend=="cuda" {"b2_train_cuda_v5r9"} else {"b2_train_metal_v5r9"})?;
    let function = unsafe{dlsym(handle,symbol.as_ptr())}; ensure(!function.is_null(),"dynamic GPU ABI symbol missing")?;
    type Kernel = unsafe extern "C" fn(*const u64,*const f32,*const f32,*const f32,*const f32,*const u64,
        *mut f32,*mut f32,*mut f32,*mut f32,*mut f32,*mut f32,*mut f32,*mut c_char,usize)->i32;
    let kernel:Kernel = unsafe{std::mem::transmute(function)};
    let active=(0..t).map(|visit|spec.actions.iter().filter(|a|itinerary.enabled(visit,a)
        && !a.mask.is_some_and(|[bank,k]|plan.masks[bank][k]==0.0)).count()).sum::<usize>();
    let cuts=itinerary.frames.iter().filter(|f|f.is_some_and(|f|plan.tbptt_window.is_some_and(|w|f>0&&f%w==0))).count();
    let dispatch_count = |reverse:bool| if mpi.is_some(){3+2*active+if reverse {t+2*active+cuts}else{0}}else{1};
    let mut gpu_dispatches=0usize;
    let mut run = |meta:&[u64],params:&[f32],tape:&mut [f32],spikes:&mut [f32],live:&mut [f32],
        grad:&mut [f32],adj:&mut [f32],logits:&mut [f32],losses:&mut [f32]| -> Result<()> {
        let code=unsafe{kernel(meta.as_ptr(),params.as_ptr(),x.as_ptr(),w.as_ptr(),v.as_ptr(),y.as_ptr(),tape.as_mut_ptr(),
            spikes.as_mut_ptr(),live.as_mut_ptr(),grad.as_mut_ptr(),adj.as_mut_ptr(),logits.as_mut_ptr(),losses.as_mut_ptr(),error.as_mut_ptr(),error.len())};
        ensure(code==0,&format!("native dynamic GPU failed: {}",unsafe{CStr::from_ptr(error.as_ptr())}.to_string_lossy()))?;
        ensure(spikes.iter().chain(grad.iter()).chain(adj.iter()).chain(logits.iter()).chain(losses.iter()).all(|x| x.is_finite())
            && live.iter().enumerate().all(|(k,x)| meta[meta[28] as usize+k%width]!=0 || x.is_finite()),"nonfinite dynamic GPU result")?;
        gpu_dispatches=gpu_dispatches.checked_add(dispatch_count(meta[9]!=0)).ok_or("GPU dispatch count overflow")?;
        Ok(())
    };
    if boundary {
        // A collection call evaluates the full baseline and probes reached
        // rates on device without ordinary reverse gradients or any commit.
        meta[9]=2;
        run(&meta,&params,&mut tape,&mut spikes,&mut live,&mut grad,&mut adj,&mut logits,&mut losses)?;
        let mut needed=Vec::with_capacity(b*t*poisson_actions);
        for sample in 0..b {for visit in 0..t {for (a,action) in spec.actions.iter().enumerate() {
            if !action_poisson(action){continue;}
            let h=meta[13] as usize+16*a;
            let at=(sample*t+visit)*tape_width+meta[h] as usize+action.reads.len()
                + if action.indirect.is_some(){action.reads.len()+2*action.writes.len()}else{0};
            let value=tape[at];
            ensure(value.is_finite()&&value>=0.0&&value<=65535.0&&value==value.floor(),"invalid GPU boundary replay mask")?;
            needed.push(value as f64);
        }}}
        // Only the owner wrote each mask. All ranks obtain the same ordered
        // replay list before launching any further GPU or MPI kernel calls.
        if let Some(m)=mpi {m.sum(&mut needed)?;}
        let mut cursor=0;
        for sample in 0..b {for visit in 0..t {for (a,action) in spec.actions.iter().enumerate() {
            if !action_poisson(action){continue;}
            let mask=needed[cursor];cursor+=1;
            ensure(mask.is_finite()&&mask>=0.0&&mask<=65535.0&&mask==mask.floor(),"invalid distributed GPU boundary replay mask")?;
            let mask=mask as u16;
            let h=meta[13] as usize+16*a;
            let noise=meta[24] as usize+(sample*t+visit)*noise_width+meta[h+12] as usize;
            for stream in 0..16 {if mask&(1<<stream)==0 {continue;}
                // Reuse admitted staging. Each complete device call starts
                // from original initial state; full batches keep counter IDs.
                meta[9]=0;let ctl=meta[14] as usize;
                meta[ctl+5]=noise as u64+1;meta[ctl+6]=stream as u64;
                run(&meta,&params,&mut tape,&mut spikes,&mut live,&mut grad,&mut adj,&mut logits,&mut losses)?;
                params[noise+80+stream]=losses[sample];
                params[noise+96]=((params[noise+96] as u16)|(1<<stream)) as f32;
            }
        }}}
        let ctl=meta[14] as usize;meta[ctl+5]=0;meta[ctl+6]=0;meta[9]=1;
    }
    // Recompute baseline forward/reverse with on-device weak loss seeds.
    // Only its final state, ordinary adjoints and optimizer are returned.
    run(&meta,&params,&mut tape,&mut spikes,&mut live,&mut grad,&mut adj,&mut logits,&mut losses)?;
    let mut gradients = state.weights.iter().map(|w| vec![0.;w.len()]).collect::<Vec<_>>();
    for (bank,row) in gradients.iter_mut().enumerate() { for (k,g) in row.iter_mut().enumerate() {
        for sample in 0..b { *g += grad[sample*edges+parameter_offsets[bank]+k] as f64 * plan.masks[bank][k]; }
    }}
    if let Some(m) = mpi { for row in &mut gradients { m.sum(row)?; } }
    if operation=="train" {apply_optimizer_distributed(plan,&mut state,&gradients,mpi)?;}
    let rows = |values:&[f32],w:usize| values.chunks_exact(w).map(|r| r.iter().map(|&v| v as f64).collect::<Vec<_>>()).collect::<Vec<_>>();
    let voltages = |values:&[f32]| values.chunks_exact(width).map(|r| spec.voltage.iter().map(|&k|r[k] as f64).collect()).collect();
    let flat:Vec<_>=spikes.iter().map(|&x|x as f64).collect();
    let (spikes,event_visits)=itinerary::outputs(spec,itinerary,&flat,b,frames);
    let poisson_state=cache.as_ref().map(|c|c.checkpoint(plan,sequence,&tape,t)).transpose()?;
    Ok(Output { poisson_state, event_visits, clock_state: None, updated_dynamic: None,schema:"b2-lif-training-result-v1",state,loss:losses.iter().map(|&v|v as f64).sum::<f64>()/b as f64,
        backend:if plan.backend=="cuda" {"cuda"} else {"metal"},
        numeric_profile:match(plan.backend.as_str(),mpi.is_some()) {
            ("cuda",true)=>"native-cuda-mpi-dynamic-actions-f32-owner-host-optimizer-f64",
            ("cuda",false)=>"native-cuda-dynamic-actions-f32-host-optimizer-f64",
            (_,true)=>"native-metal-mpi-dynamic-actions-f32-owner-host-optimizer-f64",
            (_,false)=>"native-metal-dynamic-actions-f32-host-optimizer-f64"},
        gpu_dispatches,
        gradients,initial_gradients:voltages(&adj),final_membrane:voltages(&live),
        initial_state_gradients:Some(rows(&adj,width)),final_state:Some(live.chunks_exact(width).map(|r| r.iter().enumerate()
            .map(|(k,&x)| if meta[meta[28] as usize+k]!=0 { x.to_bits() as i32 as f64 } else { x as f64 }).collect()).collect()),
        spikes,logits:rows(&logits,classes),tape_bytes:bytes+gpu_bytes,
        gradient_scope:if plan.tbptt_window.is_some_and(|w|w<frames){"tbptt-detach-boundaries"}else{"full-bptt"},
        final_tick:None,noise_sequence:None})
}
