//! Shared native Metal/CUDA forward/VJP ABI. The optimizer remains native Rust on the host.
use super::*;

#[cfg(not(unix))]
pub(super) fn execute(
    _: &Plan,
    _: State,
    _: &[Vec<Vec<f64>>],
    _: &[usize],
    _: &[Vec<f64>],
    _: usize,
    _: &str,
    _: Option<&mpi::Context>,
    _: u64,
    _: u64,
    _: Option<&poisson_cache::Checkpoint>,
) -> Result<Output> {
    Err("native GPU training requires a Unix host and the selected GPU".into())
}

#[cfg(unix)]
pub(super) fn execute(
    plan: &Plan,
    mut state: State,
    inputs: &[Vec<Vec<f64>>],
    labels: &[usize],
    initial: &[Vec<f64>],
    bytes: usize,
    operation: &str,
    mpi: Option<&mpi::Context>,
    start_tick: u64,
    noise_sequence: u64,
    poisson_state:Option<&poisson_cache::Checkpoint>,
) -> Result<Output> {
    use std::ffi::{c_char, c_void, CStr, CString};
    #[cfg_attr(target_os = "macos", link(name = "System"))]
    #[cfg_attr(not(target_os = "macos"), link(name = "dl"))]
    extern "C" {
        fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
    }
    type Kernel = unsafe extern "C" fn(
        *const u64,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const u64,
        *mut f32,
        *mut f32,
        *mut f32,
        *mut f32,
        *mut f32,
        *mut f32,
        *mut f32,
        *mut c_char,
        usize,
    ) -> i32;
    let library = CString::new(
        std::env::var(if plan.backend == "cuda" {
            "B2_TRAIN_CUDA_LIB"
        } else {
            "B2_TRAIN_METAL_LIB"
        })
        .map_err(|_| "set the selected B2_TRAIN_METAL_LIB/B2_TRAIN_CUDA_LIB native library")?,
    )?;
    let handle = unsafe { dlopen(library.as_ptr(), 2) };
    ensure(
        !handle.is_null(),
        "cannot load native Metal training library",
    )?;
    struct Library(*mut c_void);
    impl Drop for Library {
        fn drop(&mut self) {
            unsafe {
                dlclose(self.0);
            }
        }
    }
    let _library = Library(handle);
    // Appended math opcodes require this capability even when legacy layout
    // symbols match; an old shader must never silently treat them as zero.
    let math_symbol=CString::new("b2_train_math_v1")?;
    let math_function=unsafe{dlsym(handle,math_symbol.as_ptr())};
    ensure(!math_function.is_null(),"GPU math ABI capability missing")?;
    let math_version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(math_function)};
    ensure(unsafe{math_version()}==1,"GPU math ABI capability mismatch")?;

    if (0..plan.sizes.len()-1).any(|l|static_poisson::programs(plan,l).any(equation::bitwise_used)) {
        let name=CString::new("b2_train_bitwise_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU bitwise capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU bitwise capability mismatch")?;
    }

    if (0..plan.sizes.len()-1).any(|l|static_poisson::programs(plan,l).any(equation::sequence_used)) {
        let name=CString::new("b2_train_sequence_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU sequence capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU sequence capability mismatch")?;
    }
    if (0..plan.sizes.len()-1).any(|l|static_poisson::programs(plan,l).any(equation::eager_boolean_used)) {
        let name=CString::new("b2_train_boolean_eager_v1")?;let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU eager Boolean capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU eager Boolean capability mismatch")?;
    }
    // Version the metadata ABI: a v1 library cannot silently execute a v2
    // projection plan as if it were a dense feedforward network.
    let vector = plan.state_equations.is_some();
    let contextual = plan.equations.as_ref().is_some_and(|programs|
        plan.clock.is_some() || plan.noise_streams.is_some() || programs.iter().flatten().any(|node| {
            let (row,_)=equation::gpu_node(node,&vec![0;plan.masks.len()+1]);
            row[0]>=15 && row[0]!=52
        }));
    if contextual {
        let name=CString::new("b2_train_scalar_context_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU scalar context capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU scalar context capability mismatch")?;
    }
    let shared=static_poisson::used(plan);
    let boundary=shared && operation!="evaluate" && static_poisson::boundary_possible(plan);
    if shared {
        let name=CString::new("b2_train_static_poisson_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU static Poisson capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU static Poisson capability mismatch")?;
    }
    let timed_nodes = (0..plan.sizes.len()-1).flat_map(|l|static_poisson::programs(plan,l)).flatten()
        .filter(|node| matches!(node, equation::Node::TimedParameter { .. })).count();
    if timed_nodes != 0 {
        let name=CString::new("b2_train_static_timed_input_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU static timed input capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU static timed input capability mismatch")?;
    }
    let phased = vector || mpi.is_some() || shared;
    let activity = operation != "evaluate" && (vector || plan.equations.is_some());
    if activity {
        let name=CString::new("b2_train_static_vjp_activity_v1")?;
        let function=unsafe{dlsym(handle,name.as_ptr())};
        ensure(!function.is_null(),"GPU static VJP activity capability missing")?;
        let version:unsafe extern "C" fn()->u64=unsafe{std::mem::transmute(function)};
        ensure(unsafe{version()}==1,"GPU static VJP activity capability mismatch")?;
    }
    let symbol = CString::new(if vector {
        if plan.backend == "cuda" {
            "b2_train_cuda_v4r7"
        } else {
            "b2_train_metal_v4r7"
        }
    } else if mpi.is_some() || shared {
        if plan.backend == "cuda" {
            "b2_train_cuda_mpi_v1"
        } else {
            "b2_train_metal_mpi_v1"
        }
    } else {
        match (plan.backend.as_str(), plan.equations.is_some()) {
            ("cuda", true) => "b2_train_cuda_v3",
            ("cuda", false) => "b2_train_cuda_v2",
            (_, true) => "b2_train_metal_v3",
            (_, false) => "b2_train_metal_v2",
        }
    })?;
    let function = unsafe { dlsym(handle, symbol.as_ptr()) };
    ensure(
        !function.is_null(),
        "native Metal/CUDA training symbol missing",
    )?;
    let kernel: Kernel = unsafe { std::mem::transmute(function) };
    let batch = inputs.len();
    let time = inputs[0].len();
    let layers = plan.sizes.len() - 1;
    let (_, n) = plan.validate()?;
    let width = plan.state_width();
    let edges = state.weights.iter().map(Vec::len).sum::<usize>();
    let graph_edges = plan
        .projections
        .as_ref()
        .map_or(0, |ps| ps.iter().map(|p| p.sources.len()).sum::<usize>());
    // Metal has additional batch-private gradients, tape copies and scratch.
    // Both host f32 staging and Metal shared buffers remain live during the
    // dispatch. Include both copies, all three scratch arrays and metadata.
    let scalar_nodes = plan
        .equations
        .as_ref()
        .map_or(0, |ps| ps.iter().map(Vec::len).sum::<usize>());
    let vector_states = plan
        .state_equations
        .as_ref()
        .map_or(0, |ps| ps.iter().map(Vec::len).sum::<usize>());
    let equation_nodes = scalar_nodes
        + plan
            .state_equations
            .iter()
            .chain(plan.state_resets.iter())
            .flatten()
            .flatten()
            .map(Vec::len)
            .sum::<usize>();
    let ordinary_tape_count = batch
        * time
        * if vector {
            2 * width + n
        } else {
            n * if plan.equations.is_some() {
                3
            } else if plan.projections.is_some() {
                2
            } else {
                1
            }
        };
    // Host staging + device buffers, per-sample scratch, metadata and the
    // largest host-staged collective (CUDA also needs a float32 copy).
    let cache=if shared {Some(static_poisson_gpu::Layout::build(plan,start_tick,time,batch,ordinary_tape_count,poisson_state)?)}else{None};
    let cache_words=cache.as_ref().map_or(0,|c|c.words);
    let weak_words=if boundary {batch*time*n*2}else{0};
    let weak_base=ordinary_tape_count+cache_words;
    let tape_count=weak_base+weak_words;
    let noise_width = if shared {226*n}else{plan.noise_streams.as_ref().map_or(0, |counts| counts.iter().zip(&plan.sizes[1..]).map(|(c, n)| c * n).sum::<usize>())};
    let noise_count = batch * time * noise_width;
    ensure(!shared || noise_count.checked_add(2*layers+3+equation_nodes+time+edges+3*poisson_state.map_or(0,|c|c.entries.len()))
        .is_some_and(|n|n<=u32::MAX as usize),"static GPU Poisson uint32 noise address exceeded")?;
    let gpu_bytes = (2 * tape_count
        + 2 * batch * time * n
        + batch * width * 12
        + batch * edges * 2
        + edges * 2
        + batch * time * plan.sizes[0] * 2
        + batch * (plan.sizes[layers] * 2 + 2))
        * 4
        + mpi.map_or(0, |m| width * 8 * (m.size + 3) + 128)
        + graph_edges * 64
        + 84 * 16
        + equation_nodes * 80
        + timed_nodes * 64
        + layers * 128
        + vector_states * 96
        + batch * if activity || shared {16384} else {128 * 16}
        + if activity || shared {edges * 8} else {0}
        + if plan.clock.is_some() { time * 8 } else { 0 }
        + noise_count * 8 + if noise_width > 0 { (2 + 2 * layers) * 16 } else { 0 }
        + if shared {176+poisson_state.map_or(0,|c|c.entries.len())*24+weak_words*8+cache.as_ref().unwrap().host_bytes}else{0};
    ensure(
        bytes
            .checked_add(gpu_bytes)
            .is_some_and(|total| total <= plan.max_tape_bytes),
        if plan.backend == "cuda" {
            "native CUDA tape budget exceeded"
        } else {
            "native Metal tape budget exceeded"
        },
    )?;
    let equation_meta = if plan.equations.is_some() {
        3 * layers + 4 * scalar_nodes + 4 * timed_nodes
    } else {
        0
    };
    let threshold_meta = if plan.threshold_parameters.is_some() {
        if vector { n } else { layers }
    } else {
        0
    };
    // Reserve the complete v3 layout before appending: growing an already
    // large graph Vec could otherwise double its unaccounted host capacity.
    let mut meta = Vec::with_capacity(
        84 + graph_edges * 4
            + equation_meta
            + threshold_meta
            + if phased { if shared {7}else{4} } else { 0 }
            + if vector {
                1 + 6 * layers + 6 * vector_states + 4 * equation_nodes+4*timed_nodes
            } else {
                0
            }+if noise_width>0 {2+2*layers}else{0}+if shared {8}else{0},
    );
    meta.resize(84, 0u64);
    meta[79]=u64::from(contextual);
    for (i, value) in [
        batch,
        time,
        layers,
        plan.sizes[0],
        n,
        edges,
        usize::from(plan.reset == "zero"),
        usize::from(plan.detach_reset),
        plan.tbptt_window.unwrap_or(time),
        usize::from(operation != "evaluate"),
    ]
    .iter()
    .enumerate()
    {
        meta[i] = *value as u64;
    }
    if activity {meta[9] |= 2;} // legacy reverse flag + canonical mask suffix
    for l in 0..=layers {
        meta[16 + l] = plan.sizes[l] as u64;
    }
    let mut neuron = 0;
    for l in 0..layers {
        meta[40 + l] = neuron as u64;
        neuron += plan.sizes[l + 1];
    }
    meta[40 + layers] = neuron as u64;
    let mut parameter_offsets = vec![0usize];
    for weights in &state.weights {
        parameter_offsets.push(parameter_offsets.last().unwrap() + weights.len());
    }
    if let Some(projections) = &plan.projections {
        meta[10] = graph_edges as u64;
        for (q, p) in projections.iter().enumerate() {
            for e in 0..p.sources.len() {
                let source = if p.source_layer == 0 {
                    p.sources[e]
                } else {
                    meta[39 + p.source_layer] as usize + p.sources[e]
                };
                let target = meta[39 + p.target_layer] as usize + p.targets[e];
                meta.extend_from_slice(&[
                    u64::from(p.source_layer == 0),
                    source as u64,
                    target as u64,
                    (parameter_offsets[q] + p.parameter_ids[e]) as u64,
                ]);
            }
        }
    } else {
        for (l, &offset) in parameter_offsets.iter().enumerate() {
            meta[64 + l] = offset as u64;
        }
    }
    fn cast(values: impl IntoIterator<Item = f64>, count: usize) -> Result<Vec<f32>> {
        let mut converted = Vec::with_capacity(count);
        for v in values {
            let value = v as f32;
            ensure(value.is_finite(), "native GPU float32 input overflow")?;
            converted.push(value);
        }
        Ok(converted)
    }
    let mut equation_constants = Vec::with_capacity(equation_nodes);
    if let Some(programs) = &plan.equations {
        meta[11] = meta.len() as u64;
        let header = meta.len();
        meta.resize(header + 3 * layers, 0);
        for (l, program) in programs.iter().enumerate() {
            let mut descriptor=meta.len() as u64;
            for node in program {
                if let equation::Node::TimedParameter{bank,rows,columns,k,epsilon,..}=*node {
                    ensure((epsilon as f32).is_finite() && (epsilon as f32)>0.,
                        "TimedArray time resolution underflows float32")?;
                    ensure(rows<=1<<24 && columns<=1<<24,
                        "GPU static TimedArray dimensions exceed exact float32 index range")?;
                    meta.extend_from_slice(&[parameter_offsets[bank] as u64,rows as u64,columns as u64,k]);
                }
            }
            let poisson=program.iter().any(|n|matches!(n,equation::Node::Poisson{..}));
            meta[header + 3 * l] = program.len() as u64 | if poisson {256|1024|if boundary {512}else{0}}else{0};
            meta[header + 3 * l + 1] = meta.len() as u64;
            meta[header + 3 * l + 2] = (2 * layers + 3 + equation_constants.len()) as u64;
            for node in program {
                let (mut row, value) = equation::gpu_node(node, &parameter_offsets);
                if let equation::Node::Poisson{rate,..}=*node {row[3]=poisson_ir::rate_differentiable(program,rate) as u64;}
                if matches!(node,equation::Node::TimedParameter{..}) {row[3]=descriptor;descriptor+=4;}
                meta.extend_from_slice(&row);
                equation_constants.push(value);
            }
        }
    }
    if let Some(updates) = &plan.state_equations {
        let resets = plan.state_resets.as_ref().unwrap();
        meta[83] = meta.len() as u64;
        let header = meta.len();
        meta.resize(header + 1 + 6 * layers, 0);
        meta[header] = width as u64;
        let mut state_offset = 0;
        for l in 0..layers {
            let h = header + 1 + 6 * l;
            meta[h] = updates[l].len() as u64;
            meta[h + 1] = state_offset as u64;
            if let Some(spec) = plan.refractory.as_ref().and_then(|r| r[l].as_ref()) {
                meta[h + 4] = spec.steps as u64 + 1;
                meta[h + 5] = spec.clamp.iter().fold(0, |mask, &s| mask | (1u64 << s));
            }
            state_offset += updates[l].len() * plan.sizes[l + 1];
            for (kind, programs) in [&updates[l], &resets[l]].iter().enumerate() {
                let headers = meta.len();
                meta[h + 2 + kind] = headers as u64;
                meta.resize(headers + 3 * programs.len(), 0);
                for (i, program) in programs.iter().enumerate() {
                    let mut descriptor=meta.len() as u64;
                    for node in program {
                        if let equation::Node::TimedParameter {bank,rows,columns,k,epsilon,..}=*node {
                            ensure((epsilon as f32).is_finite() && (epsilon as f32)>0.,
                                "TimedArray time resolution underflows float32")?;
                            ensure(rows<=1<<24 && columns<=1<<24,
                                "GPU static TimedArray dimensions exceed exact float32 index range")?;
                            meta.extend_from_slice(&[parameter_offsets[bank] as u64,rows as u64,columns as u64,k]);
                        }
                    }
                    let poisson=program.iter().any(|n|matches!(n,equation::Node::Poisson{..}));
                    meta[headers + 3 * i] = program.len() as u64 | if poisson {256|1024|if boundary {512}else{0}}else{0};
                    meta[headers + 3 * i + 1] = meta.len() as u64;
                    meta[headers + 3 * i + 2] = (2 * layers + 3 + equation_constants.len()) as u64;
                    for node in program {
                        let (mut row, value) = equation::gpu_node(node, &parameter_offsets);
                        if let equation::Node::Poisson{rate,..}=*node {row[3]=poisson_ir::rate_differentiable(program,rate) as u64;}
                        if matches!(node,equation::Node::TimedParameter{..}) {
                            row[3]=descriptor; descriptor+=4;
                        }
                        meta.extend_from_slice(&row);
                        equation_constants.push(value);
                    }
                }
            }
        }
    }
    if let Some(references) = &plan.threshold_parameters {
        meta[14] = meta.len() as u64;
        for l in 0..references.len() {
            for j in 0..if vector { plan.sizes[l + 1] } else { 1 } {
                meta.push(plan.threshold_reference(l, j).map_or(0, |[bank, index]| {
                    (parameter_offsets[bank] + index + 1) as u64
                }));
            }
        }
    }
    if phased {
        meta[15] = meta.len() as u64;
        meta.extend_from_slice(&[
            mpi.map_or(1, |m| m.size) as u64,
            mpi.map_or(0, |m| m.rank) as u64,
            0,
            0,
        ]);
        if shared {meta.extend_from_slice(&[0,0,0]);}
    }
    if let Some(m) = mpi {
        use sha2::{Digest, Sha256};
        m.agree(&Sha256::digest(std::fs::read(library.to_str()?)?).into())?;
    }
    meta[12] = meta.len() as u64;
    meta[13] = (2 * layers + 3 + equation_constants.len()) as u64;
    let mut params = cast(
        [
            plan.beta.clone(),
            plan.threshold.clone(),
            vec![plan.surrogate.slope, plan.surrogate.scale, plan.logit_scale],
        ]
        .concat(),
        2 * layers + 3,
    )?;
    params.extend(cast(equation_constants, equation_nodes)?);
    if plan.clock.is_some() {
        meta[82] = params.len() as u64;
        let times = cast((0..time).map(|t| plan.time_at(start_tick + t as u64)), time)?;
        ensure(times.windows(2).all(|w| w[1] > w[0])
            && (plan.time_at(start_tick + time as u64) as f32).is_finite()
            && (plan.time_at(start_tick + time as u64) as f32) > times[time - 1],
            "GPU clock has insufficient float32 precision")?;
        params.extend(times);
        meta[13] = params.len() as u64;
    }
    if let Some(counts) = &plan.noise_streams {
        meta[81] = meta.len() as u64;
        meta.reserve_exact(2 + 2 * layers);
        meta.push(params.len() as u64);
        meta.push(noise_width as u64);
        let mut offset = 0;
        for l in 0..layers {
            let stride=if shared {226}else{counts[l]};
            meta.extend_from_slice(&[stride as u64, offset as u64]);
            offset += stride * plan.sizes[l + 1];
        }
        params.reserve_exact(noise_count);
        for b in 0..batch {
            for t in 0..time {
                for l in 0..layers {
                    let uniform = equation::noise_masks(static_poisson::programs(plan,l)).1;
                    for j in 0..plan.sizes[l + 1] {
                        let values = plan.noise_at(uniform, noise_sequence, b, l, j, start_tick + t as u64);
                        if let Some(cache)=&cache {
                            let action=static_poisson::action(plan,l,j);let mut keys=[0.;80];
                            poisson_ir::encode_keys(&mut keys,plan,&action,noise_sequence,b,start_tick+t as u64);
                            for _phase in 0..2 {
                                params.extend(values.iter().enumerate().map(|(stream,&v)|noise::device_sample(v,uniform&(1<<stream)!=0)));
                                params.extend(keys[16..80].iter().map(|&v|v as f32));
                                params.resize(params.len()+17,0.);
                                for stream in 0..16 {params.push(f32::from_bits(cache.offset(poisson_cache::identity(&action,stream,b,start_tick+t as u64))));}
                            }
                        }else{
                            params.extend(values[..counts[l]].iter().enumerate().map(|(stream, &v)|
                                noise::device_sample(v, uniform & (1 << stream) != 0)));
                        }
                    }
                }
            }
        }
        meta[12] = meta.len() as u64;
        meta[13] = params.len() as u64;
    }
    if let Some(cache)=&cache {
        meta[80]=meta.len() as u64;
        let import_start=params.len();
        if let Some(state)=poisson_state {for e in &state.entries {
            params.extend_from_slice(&[f32::from_bits(cache.offset(e.identity)),f32::from_bits(e.count as u32),e.rate as f32]);
        }}
        meta.extend_from_slice(&[tape_count as u64,cache.base as u64,cache.lane_words as u64,import_start as u64,
            poisson_state.map_or(0,|c|c.entries.len()) as u64,weak_base as u64,weak_words as u64,u64::from(boundary)]);
        meta[12]=meta.len() as u64;
    }
    if activity || shared {
        params.reserve_exact(edges);
        for &mask in plan.masks.iter().flatten(){params.push(mask as f32);}
        meta[13]=params.len() as u64;
    }
    let x = cast(
        inputs.iter().flatten().flatten().copied(),
        batch * time * plan.sizes[0],
    )?;
    let w = cast(state.weights.iter().flatten().copied(), edges)?;
    let v = cast(initial.iter().flatten().copied(), batch * width)?;
    let y = labels.iter().map(|&v| v as u64).collect::<Vec<_>>();
    let mut pre = vec![0f32; tape_count];
    let mut spikes = vec![0f32; batch * time * n];
    let mut final_v = vec![0f32; batch * width];
    let mut grad = vec![0f32; batch * edges];
    let mut initial_grad = final_v.clone();
    let classes = plan.sizes[layers];
    let mut logits = vec![0f32; batch * classes];
    let mut loss = vec![0f32; batch];
    let mut error = vec![0 as c_char; 2048];
    let mut gpu_dispatches=0;
    let mut run=|meta:&[u64],params:&[f32],pre:&mut Vec<f32>,spikes:&mut Vec<f32>,final_v:&mut Vec<f32>,
        grad:&mut Vec<f32>,initial_grad:&mut Vec<f32>,logits:&mut Vec<f32>,loss:&mut Vec<f32>|->Result<()> {
    let code = unsafe {
        kernel(
            meta.as_ptr(),
            params.as_ptr(),
            x.as_ptr(),
            w.as_ptr(),
            v.as_ptr(),
            y.as_ptr(),
            pre.as_mut_ptr(),
            spikes.as_mut_ptr(),
            final_v.as_mut_ptr(),
            grad.as_mut_ptr(),
            initial_grad.as_mut_ptr(),
            logits.as_mut_ptr(),
            loss.as_mut_ptr(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    let result=if code != 0 {
        Err(format!(
            "native GPU training failed: {}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        )
        .into())
    }else{ensure(
        spikes
            .iter()
            .chain(final_v.iter())
            .chain(grad.iter())
            .chain(initial_grad.iter())
            .chain(logits.iter())
            .chain(loss.iter())
            .all(|v| v.is_finite()),
        "nonfinite native GPU training output",
    )};
    static_poisson::reconcile(result,mpi,shared)?;
    gpu_dispatches+=if phased {3+time*if meta[9]&1==0 {2}else{4}}else{1};
    Ok(())
    };
    if boundary {
        let control=meta[15] as usize;meta[control+4]=1;
        run(&meta,&params,&mut pre,&mut spikes,&mut final_v,&mut grad,&mut initial_grad,&mut logits,&mut loss)?;
        let mut needed=pre[weak_base..weak_base+weak_words].iter().map(|&v|v as f64).collect::<Vec<_>>();
        if let Some(m)=mpi {m.sum(&mut needed)?;}
        for b in 0..batch {for t in 0..time {for l in 0..layers {for j in 0..plan.sizes[l+1] {for phase in 0..2 {
            let at=((b*time+t)*n+meta[40+l] as usize+j)*2+phase;
            let mask=needed[at];ensure(mask.is_finite()&&mask>=0.&&mask<=65535.&&mask==mask.floor(),"invalid static GPU weak replay mask")?;
            let noise=meta[meta[81] as usize] as usize+(b*time+t)*noise_width
                +meta[meta[81] as usize+3+2*l] as usize+j*226+phase*113;
            for stream in 0..16 {if (mask as u16)&(1<<stream)!=0 {
                meta[9]=0;meta[control+4]=0;meta[control+5]=noise as u64+1;meta[control+6]=stream as u64;
                run(&meta,&params,&mut pre,&mut spikes,&mut final_v,&mut grad,&mut initial_grad,&mut logits,&mut loss)?;
                params[noise+80+stream]=loss[b];params[noise+96]=((params[noise+96] as u16)|(1<<stream)) as f32;
            }}
        }}}}}
        meta[9]=3;meta[control+4]=0;meta[control+5]=0;meta[control+6]=0;
    }
    run(&meta,&params,&mut pre,&mut spikes,&mut final_v,&mut grad,&mut initial_grad,&mut logits,&mut loss)?;
    drop(run);
    let checkpoint=cache.as_ref().map(|c|c.checkpoint(plan,noise_sequence,&pre,mpi)).transpose()?;
    let mut gradients = state
        .weights
        .iter()
        .map(|w| vec![0.0; w.len()])
        .collect::<Vec<_>>();
    for l in 0..gradients.len() {
        for e in 0..gradients[l].len() {
            for b in 0..batch {
                gradients[l][e] +=
                    grad[b * edges + parameter_offsets[l] + e] as f64 * plan.masks[l][e];
            }
        }
    }
    if let Some(m) = mpi {
        for row in &mut gradients {
            m.sum(row)?;
        }
    }
    if operation == "train" {
        apply_optimizer_distributed(plan, &mut state, &gradients, mpi)?;
    }
    let rows = |values: &[f32], width: usize| {
        values
            .chunks_exact(width)
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect()
    };
    let rows: fn(&[f32], usize) -> Vec<Vec<f64>> = rows;
    let voltage_rows = |values: &[f32]| {
        let all = rows(values, width);
        if !vector {
            return all;
        }
        all.iter()
            .map(|row| {
                (0..layers)
                    .flat_map(|l| {
                        let offset = meta[meta[83] as usize + 2 + 6 * l] as usize;
                        row[offset..offset + plan.sizes[l + 1]].iter().copied()
                    })
                    .collect()
            })
            .collect()
    };
    let shaped = spikes
        .chunks_exact(time * n)
        .map(|sample| rows(sample, n))
        .collect();
    Ok(Output { poisson_state: checkpoint, event_visits: None, clock_state: None, updated_dynamic: None,
        final_tick: None,
        noise_sequence: None,
        schema: "b2-lif-training-result-v1",
        state,
        loss: loss.iter().map(|&v| v as f64).sum::<f64>() / batch as f64,
        backend: if plan.backend == "cuda" {
            "cuda"
        } else {
            "metal"
        },
        numeric_profile: if vector {
            match (plan.backend.as_str(), mpi.is_some()) {
                ("cuda", true) => "native-cuda-mpi-multistate-target-owned-f32-host-optimizer-f64",
                ("cuda", false) => "native-cuda-multistate-f32-host-optimizer-f64",
                (_, true) => "native-metal-mpi-multistate-target-owned-f32-host-optimizer-f64",
                (_, false) => "native-metal-multistate-f32-host-optimizer-f64",
            }
        } else if mpi.is_some() && plan.backend == "cuda" {
            "native-cuda-mpi-target-owned-f32-host-optimizer-f64"
        } else if mpi.is_some() {
            "native-metal-mpi-target-owned-f32-host-optimizer-f64"
        } else if plan.backend == "cuda" {
            "native-cuda-forward-backward-f32-host-optimizer-f64"
        } else {
            "native-metal-forward-backward-f32-host-optimizer-f64"
        },
        gpu_dispatches,
        gradients,
        initial_gradients: voltage_rows(&initial_grad),
        final_membrane: voltage_rows(&final_v),
        final_state: vector.then(|| rows(&final_v, width)),
        initial_state_gradients: vector.then(|| rows(&initial_grad, width)),
        spikes: shaped,
        logits: rows(&logits, classes),
        tape_bytes: bytes + gpu_bytes,
        gradient_scope: if plan.tbptt_window.is_some_and(|w| w < time) {
            "tbptt-detach-boundaries"
        } else {
            "full-bptt"
        },
    })
}
