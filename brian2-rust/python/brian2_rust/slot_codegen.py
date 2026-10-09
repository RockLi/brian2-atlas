"""Canonical CPU schedule emission for models outside the fixed-phase plan.

Each logical node becomes compiled Rust statements. This is an AOT path, not
an invocation of the reference interpreter. It deliberately keeps node order
and uses serial ownership until a separate physical optimization proves safety.
"""


def initialize_queues(model, *, distributed=False):
    lines = []
    for q, (definition, instance) in enumerate(zip(
            model["definition"]["synapses"], model["instance"]["synapses"], strict=True)):
        for r, pathway in enumerate(instance["pathways"]):
            prefix = f"s{q}_" if r == 0 else f"s{q}p{r}_"
            queue = f"plan_s{q}p{r}_queue"
            delays = pathway["delay_ticks"]
            uniform = bool(delays) and all(d == delays[0] for d in delays)
            maximum = f"{prefix}delay_ticks" if uniform else f"{prefix}delay_ticks.iter().copied().max().unwrap_or(0)"
            endpoint = "source" if pathway["kind"] == "pre" else "target"
            p = definition[f"{endpoint}_population"]
            steps = model["definition"]["populations"][p]["steps"]
            lines += [f"    let mut {queue}: Vec<Vec<usize>> = vec![Vec::new(); (({maximum})+1).min({steps}usize.max(1))];"]
            if distributed and instance['topology']['kind'] == 'fixed_total':
                # Procedural v1 has no pending arrays or continuation contract.
                continue
            if distributed and not pathway['pending']:
                continue
            lines += [
                      f"    for (&delivery, &item) in {prefix}pending_ticks.iter().zip(&{prefix}pending_items) {{",
                      f"        let slot = delivery % {queue}.len();"]
            if uniform:
                if distributed and endpoint == "source":
                    lines.append(f"        if s{q}_local_edge_count != 0 && s{q}_offsets[item] != s{q}_offsets[item+1] {{ {queue}[slot].push(item); }}")
                else:
                    lines += _edges_for_endpoint(model, q, endpoint, "item", indent="        ", distributed=distributed)
                    lines.append(f"            {queue}[slot].push(edge);")
                    lines.append("        }")
            else:
                lines.append(f"        {queue}[slot].push(item);")
            lines.append("    }")
    return lines


def _edges_for_endpoint(model, q, endpoint, index, indent="            ", distributed=False):
    from .planner import _explicit_topology
    if endpoint == "target":
        return [f"{indent}for &edge_id in &s{q}_target_edges[s{q}_target_offsets[{index}]..s{q}_target_offsets[{index}+1]] {{",
                f"{indent}    let edge = edge_id as usize;"]
    if not distributed and _explicit_topology(model["instance"]["synapses"][q]):
        return [f"{indent}for &edge_id in &s{q}_edges[s{q}_offsets[{index}]..s{q}_offsets[{index}+1]] {{",
                f"{indent}    let edge = edge_id as usize;"]
    return [f"{indent}for edge in s{q}_offsets[{index}]..s{q}_offsets[{index}+1] {{"]


def _endpoint_requirements(synapse, code, *, reduction_target=None):
    """Return endpoint identities that observable code actually requires.

    Distributed CSR shards do not store a direct source index. Reconstructing
    it with ``partition_point`` inside every edge/tick loop is substantial work,
    so an edge-local CodeObject must not pay for endpoints it never reads or
    writes. The reduction endpoint is physical addressing and is required even
    though it is not listed among expression effects.
    """
    names = set(code["effects"]["reads"]) | set(code["effects"]["writes"])
    source = bool(names & ({"i"} | set(synapse["pre_state_aliases"])))
    target = bool(names & ({"j", "not_refractory_post", "lastspike_post"}
                           | set(synapse["post_state_aliases"])))
    return source or reduction_target == "pre", target or reduction_target == "post"


def _endpoints(model, q, *, distributed=False, need_source=True, need_target=True):
    from .planner import _explicit_topology
    definition = model["definition"]["synapses"][q]
    source = (f"s{q}_source_index[edge] as usize" if not distributed and _explicit_topology(model["instance"]["synapses"][q])
              else f"s{q}_offsets.partition_point(|&offset| offset <= edge) - 1")
    lines = []
    if need_source:
        lines += [f"let source = {source};",
                  f"let source_state = source + {definition['source_start']};"]
    if need_target:
        lines += [f"let target = s{q}_target_index[edge] as usize;",
                  f"let target_state = target + {definition['target_start']};"]
    return lines


def _synapse_writes(model, q, code, symbols):
    from .native import _storage_store
    synapse = model["definition"]["synapses"][q]
    populations = model["definition"]["populations"]
    states = {s["name"]: (i, s) for i, s in enumerate(synapse["states"])}
    lines = []
    for name in code["effects"]["writes"]:
        if name in states:
            index, symbol = states[name]
            lines.append(f"s{q}_state_{index}[edge] = {_storage_store(symbol, symbols[name])};")
            continue
        for side, endpoint in (("pre", "source"), ("post", "target")):
            if name in synapse[f"{side}_state_aliases"]:
                p = synapse[f"{endpoint}_population"]
                state = synapse[f"{side}_state_aliases"][name]
                index, symbol = next((i, s) for i, s in enumerate(populations[p]["states"]) if s["name"] == state)
                lines.append(f"p{p}_state_{index}[{endpoint}_state] = {_storage_store(symbol, symbols[name])};")
                break
        else:
            raise ValueError(f"unsupported canonical pathway write {name}")
    return lines


def emit_schedule(model, plan, *, needs_event_dump, distributed=False, gpu_offload=False):
    from .native import (_v7_block, _storage_store, _v7_event_vector,
                         _v7_inputs, _v7_synapse_link_value, _number)
    d = model["definition"]
    summed_caches = ({} if distributed else
                     plan.cpu.choices().get("summed_endpoint_caches", {}))
    lines = []
    recorded = set()
    pending_events = []
    pending_clock = None

    def append(block):
        lines.extend("            " + line for line in block)

    def adaptive_lines(p, item, pop, code):
        """Emit a serial Fehlberg 4(5) driver for one population CodeObject."""
        if distributed or gpu_offload:
            raise ValueError("adaptive GSL integration currently requires local CPU AOT")
        adaptive = code["adaptive"]
        states = adaptive["states"]
        derivatives = adaptive["derivatives"]
        positions = {symbol["name"]: pos for pos, symbol in enumerate(pop["states"])}
        overrides = {name: f"gsl_y[{position}]"
                     for position, name in enumerate(states)}
        overrides.update(t="gsl_stage_time", dt=f"p{p}_dt")
        scalar, vector, symbols = _v7_block(
            model, code, p, "neuron", clock=pop["clock"], overrides=overrides)
        derivative_values = []
        frozen = set(adaptive["frozen_states"])
        for state, derivative in zip(states, derivatives, strict=True):
            value = symbols[derivative]
            if state in frozen:
                value = f"if p{p}_not_refractory[i] != 0 {{ {value} }} else {{ 0.0 }}"
            derivative_values.append(value)
        n = len(states)
        prefix = f"gsl_p{p}_{item}"
        state_values = ", ".join(
            f"p{p}_state_{positions[name]}[i] as f64" for name in states)
        errors = ", ".join(repr(_number(value))
                           for value in adaptive["absolute_errors"])
        last = adaptive["last_timestep"]
        initial_h = (f"p{p}_state_{positions[last]}[i] as f64"
                     if last is not None else f"p{p}_dt")
        out = [
            f"let {prefix}_initial = [{state_values}];",
            f"let mut {prefix}_y = {prefix}_initial;",
            f"let {prefix}_errors: [f64; {n}] = [{errors}];",
            f"let mut {prefix}_current = time;",
            f"let {prefix}_end = time + p{p}_dt;",
            f"let mut {prefix}_driver_h = ({initial_h}).min(p{p}_dt);",
            f"if !{prefix}_driver_h.is_finite() || {prefix}_driver_h <= 0.0 {{ {prefix}_driver_h = p{p}_dt; }}",
            f"let mut {prefix}_accepted = 0usize;",
            f"let mut {prefix}_failed = 0usize;",
            f"let mut {prefix}_eval = |gsl_y: &[f64; {n}], gsl_stage_time: f64| -> [f64; {n}] {{",
        ]
        out += ["    " + line for line in scalar + vector]
        out += [f"    [{', '.join(derivative_values)}]", "};"]
        out += [f"while {prefix}_current < {prefix}_end {{",
                f"    if {prefix}_accepted > {adaptive['max_steps']} {{ return Err(\"adaptive GSL integrator exceeded max_steps\".into()); }}",
                f"    let {prefix}_remaining = {prefix}_end - {prefix}_current;",
                f"    let {prefix}_final_step = {prefix}_driver_h > {prefix}_remaining;",
                f"    let {prefix}_step_h = {prefix}_driver_h.min({prefix}_remaining);",
                f"    if {prefix}_step_h <= 0.0 || {prefix}_current + {prefix}_step_h <= {prefix}_current {{ return Err(\"adaptive GSL timestep underflow\".into()); }}",
                f"    let {prefix}_k1 = {prefix}_eval(&{prefix}_y, {prefix}_current);",
                f"    let mut {prefix}_stage = [0.0f64; {n}];",
                f"    let mut {prefix}_candidate = [0.0f64; {n}];",
                f"    let mut {prefix}_error = 0.0f64;",
        ]

        def weighted(terms):
            return " + ".join(
                f"({coefficient!r})*{prefix}_k{stage}[q]"
                for stage, coefficient in terms) or "0.0"

        def embedded_lines(nodes, rows, high, *, error=None, low=None):
            block = []
            for stage_number, (node, row) in enumerate(
                    zip(nodes, rows, strict=True), start=2):
                block += [
                    f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*({weighted(row)}); }}",
                    f"    let {prefix}_k{stage_number} = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h*{node!r});",
                ]
            block += [f"    for q in 0..{n} {{",
                      f"        let high_sum = {weighted(high)};",
                      f"        {prefix}_candidate[q] = {prefix}_y[q] + {prefix}_step_h*high_sum;"]
            if low is not None:
                block += [f"        let low_sum = {weighted(low)};",
                          f"        {prefix}_error = {prefix}_error.max(({prefix}_step_h*(low_sum - high_sum)).abs()/{prefix}_errors[q]);"]
            else:
                block += [f"        let error_sum = {weighted(error)};",
                          f"        {prefix}_error = {prefix}_error.max(({prefix}_step_h*error_sum).abs()/{prefix}_errors[q]);"]
            block.append("    }")
            return block

        if adaptive["integrator"] == "rk2":
            order = 2.0
            out += [
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + 0.5*{prefix}_step_h*{prefix}_k1[q]; }}",
                f"    let {prefix}_k2 = {prefix}_eval(&{prefix}_stage, {prefix}_current + 0.5*{prefix}_step_h);",
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*(-{prefix}_k1[q] + 2.0*{prefix}_k2[q]); }}",
                f"    let {prefix}_k3 = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h);",
                f"    for q in 0..{n} {{",
                f"        let high_sum = ({prefix}_k1[q] + 4.0*{prefix}_k2[q] + {prefix}_k3[q])/6.0;",
                f"        {prefix}_candidate[q] = {prefix}_y[q] + {prefix}_step_h*high_sum;",
                f"        {prefix}_error = {prefix}_error.max(({prefix}_step_h*({prefix}_k2[q] - high_sum)).abs()/{prefix}_errors[q]);",
                "    }",
            ]
        elif adaptive["integrator"] == "rk4":
            order = 4.0
            out += [
                f"    let mut {prefix}_rk4 = |base: &[f64; {n}], start: f64, step: f64| -> [f64; {n}] {{",
                f"        let k1 = {prefix}_eval(base, start);",
                f"        let mut stage = [0.0f64; {n}];",
                f"        for q in 0..{n} {{ stage[q] = base[q] + 0.5*step*k1[q]; }}",
                f"        let k2 = {prefix}_eval(&stage, start + 0.5*step);",
                f"        for q in 0..{n} {{ stage[q] = base[q] + 0.5*step*k2[q]; }}",
                f"        let k3 = {prefix}_eval(&stage, start + 0.5*step);",
                f"        for q in 0..{n} {{ stage[q] = base[q] + step*k3[q]; }}",
                f"        let k4 = {prefix}_eval(&stage, start + step);",
                "        let mut result = *base;",
                f"        for q in 0..{n} {{",
                "            result[q] += step/6.0*k1[q];",
                "            result[q] += step/3.0*k2[q];",
                "            result[q] += step/3.0*k3[q];",
                "            result[q] += step/6.0*k4[q];",
                "        }",
                "        result",
                "    };",
                f"    let {prefix}_one = {prefix}_rk4(&{prefix}_y, {prefix}_current, {prefix}_step_h);",
                f"    let {prefix}_half = {prefix}_rk4(&{prefix}_y, {prefix}_current, 0.5*{prefix}_step_h);",
                f"    {prefix}_candidate = {prefix}_rk4(&{prefix}_half, {prefix}_current + 0.5*{prefix}_step_h, 0.5*{prefix}_step_h);",
                f"    for q in 0..{n} {{ {prefix}_error = {prefix}_error.max((4.0*({prefix}_candidate[q] - {prefix}_one[q])/15.0).abs()/{prefix}_errors[q]); }}",
            ]
        elif adaptive["integrator"] == "rkf45":
            order = 5.0
            out += [
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*(0.25*{prefix}_k1[q]); }}",
                f"    let {prefix}_k2 = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h*0.25);",
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*((3.0/32.0)*{prefix}_k1[q] + (9.0/32.0)*{prefix}_k2[q]); }}",
                f"    let {prefix}_k3 = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h*(3.0/8.0));",
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*((1932.0/2197.0)*{prefix}_k1[q] - (7200.0/2197.0)*{prefix}_k2[q] + (7296.0/2197.0)*{prefix}_k3[q]); }}",
                f"    let {prefix}_k4 = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h*(12.0/13.0));",
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*((439.0/216.0)*{prefix}_k1[q] - 8.0*{prefix}_k2[q] + (3680.0/513.0)*{prefix}_k3[q] - (845.0/4104.0)*{prefix}_k4[q]); }}",
                f"    let {prefix}_k5 = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h);",
                f"    for q in 0..{n} {{ {prefix}_stage[q] = {prefix}_y[q] + {prefix}_step_h*(-(8.0/27.0)*{prefix}_k1[q] + 2.0*{prefix}_k2[q] - (3544.0/2565.0)*{prefix}_k3[q] + (1859.0/4104.0)*{prefix}_k4[q] - (11.0/40.0)*{prefix}_k5[q]); }}",
                f"    let {prefix}_k6 = {prefix}_eval(&{prefix}_stage, {prefix}_current + {prefix}_step_h*0.5);",
                f"    for q in 0..{n} {{",
                f"        {prefix}_candidate[q] = {prefix}_y[q] + {prefix}_step_h*((16.0/135.0)*{prefix}_k1[q] + (6656.0/12825.0)*{prefix}_k3[q] + (28561.0/56430.0)*{prefix}_k4[q] - 0.18*{prefix}_k5[q] + (2.0/55.0)*{prefix}_k6[q]);",
                f"        let error_sum = {prefix}_k1[q]/360.0 - 128.0*{prefix}_k3[q]/4275.0 - 2197.0*{prefix}_k4[q]/75240.0 + {prefix}_k5[q]/50.0 + 2.0*{prefix}_k6[q]/55.0;",
                f"        {prefix}_error = {prefix}_error.max(({prefix}_step_h*error_sum).abs()/{prefix}_errors[q]);",
                "    }",
            ]
        elif adaptive["integrator"] == "rkck":
            order = 5.0
            out += embedded_lines(
                [1/5, 3/10, 3/5, 1.0, 7/8],
                [
                    [(1, 1/5)],
                    [(1, 3/40), (2, 9/40)],
                    [(1, 3/10), (2, -9/10), (3, 6/5)],
                    [(1, -11/54), (2, 5/2), (3, -70/27), (4, 35/27)],
                    [(1, 1631/55296), (2, 175/512), (3, 575/13824),
                     (4, 44275/110592), (5, 253/4096)],
                ],
                [(1, 37/378), (3, 250/621), (4, 125/594), (6, 512/1771)],
                error=[
                    (1, 37/378 - 2825/27648),
                    (3, 250/621 - 18575/48384),
                    (4, 125/594 - 13525/55296),
                    (5, -277/14336),
                    (6, 512/1771 - 1/4),
                ],
            )
        elif adaptive["integrator"] == "rk8pd":
            order = 8.0
            out += embedded_lines(
                [1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200,
                 5490023248/9719169821, 13/20, 1201146811/1299019798,
                 1.0, 1.0],
                [
                    [(1, 1/18)],
                    [(1, 1/48), (2, 1/16)],
                    [(1, 1/32), (3, 3/32)],
                    [(1, 5/16), (3, -75/64), (4, 75/64)],
                    [(1, 3/80), (4, 3/16), (5, 3/20)],
                    [(1, 29443841/614563906), (4, 77736538/692538347),
                     (5, -28693883/1125000000), (6, 23124283/1800000000)],
                    [(1, 16016141/946692911), (4, 61564180/158732637),
                     (5, 22789713/633445777), (6, 545815736/2771057229),
                     (7, -180193667/1043307555)],
                    [(1, 39632708/573591083), (4, -433636366/683701615),
                     (5, -421739975/2616292301), (6, 100302831/723423059),
                     (7, 790204164/839813087), (8, 800635310/3783071287)],
                    [(1, 246121993/1340847787), (4, -37695042795/15268766246),
                     (5, -309121744/1061227803), (6, -12992083/490766935),
                     (7, 6005943493/2108947869), (8, 393006217/1396673457),
                     (9, 123872331/1001029789)],
                    [(1, -1028468189/846180014), (4, 8478235783/508512852),
                     (5, 1311729495/1432422823), (6, -10304129995/1701304382),
                     (7, -48777925059/3047939560), (8, 15336726248/1032824649),
                     (9, -45442868181/3398467696), (10, 3065993473/597172653)],
                    [(1, 185892177/718116043), (4, -3185094517/667107341),
                     (5, -477755414/1098053517), (6, -703635378/230739211),
                     (7, 5731566787/1027545527), (8, 5232866602/850066563),
                     (9, -4093664535/808688257), (10, 3962137247/1805957418),
                     (11, 65686358/487910083)],
                    [(1, 403863854/491063109), (4, -5068492393/434740067),
                     (5, -411421997/543043805), (6, 652783627/914296604),
                     (7, 11173962825/925320556), (8, -13158990841/6184727034),
                     (9, 3936647629/1978049680), (10, -160528059/685178525),
                     (11, 248638103/1413531060)],
                ],
                [(1, 14005451/335480064), (6, -59238493/1068277825),
                 (7, 181606767/758867731), (8, 561292985/797845732),
                 (9, -1041891430/1371343529), (10, 760417239/1151165299),
                 (11, 118820643/751138087), (12, -528747749/2220607170),
                 (13, 1/4)],
                low=[
                    (1, 13451932/455176623), (6, -808719846/976000145),
                    (7, 1757004468/5645159321), (8, 656045339/265891186),
                    (9, -3867574721/1518517206), (10, 465885868/322736535),
                    (11, 53011238/667516719), (12, 2/45),
                ],
            )
        else:
            raise ValueError(f"unsupported adaptive integrator {adaptive['integrator']}")
        out += [
            f"    if !{prefix}_error.is_finite() || {prefix}_candidate.iter().any(|value| !value.is_finite()) {{ return Err(\"adaptive GSL integration produced a non-finite value\".into()); }}",
            f"    let {prefix}_decrease = {prefix}_error > 1.1;",
            f"    let {prefix}_factor = if {prefix}_decrease {{ (0.9/{prefix}_error.powf(1.0/{order!r})).max(0.2) }} else if {prefix}_error < 0.5 {{ if {prefix}_error <= f64::MIN_POSITIVE {{ 5.0 }} else {{ (0.9/{prefix}_error.powf(1.0/({order!r} + 1.0))).clamp(1.0, 5.0) }} }} else {{ 1.0 }};",
            f"    let {prefix}_adjusted_h = {prefix}_step_h*{prefix}_factor;",
        ]
        if not adaptive["adaptable_timestep"]:
            out += [f"    if {prefix}_decrease {{ return Err(\"fixed-step GSL integration exceeded absolute_error\".into()); }}"]
        condition = "true" if not adaptive["adaptable_timestep"] else f"!{prefix}_decrease"
        out += [f"    if {condition} {{",
                f"        {prefix}_y = {prefix}_candidate;",
                f"        {prefix}_current = if {prefix}_final_step {{ {prefix}_end }} else {{ {prefix}_current + {prefix}_step_h }};",
                f"        {prefix}_accepted += 1;",
                "    } else {",
                f"        {prefix}_failed += 1;",
                f"        {prefix}_driver_h = {prefix}_adjusted_h;",
                "    }",
        ]
        if adaptive["adaptable_timestep"]:
            out += [f"    if !{prefix}_decrease && !{prefix}_final_step {{ {prefix}_driver_h = {prefix}_adjusted_h.min(p{p}_dt); }}"]
        out += ["}"]
        for position, name in enumerate(states):
            out.append(f"p{p}_state_{positions[name]}[i] = {prefix}_y[{position}];")
        if last is not None:
            out.append(f"p{p}_state_{positions[last]}[i] = {prefix}_driver_h;")
        if adaptive["failed_steps"] is not None:
            pos = positions[adaptive["failed_steps"]]
            out.append(f"p{p}_state_{pos}[i] = {prefix}_failed as i32;")
        if adaptive["step_count"] is not None:
            pos = positions[adaptive["step_count"]]
            out.append(f"p{p}_state_{pos}[i] = {prefix}_accepted as i32;")
        return out

    def record_history(p, event):
        if needs_event_dump:
            position = d["populations"][p]["events"].index(event)
            vector = _v7_event_vector(p, d["populations"][p], event)
            append([f"if {'mpi.rank == 0 && ' if distributed else ''}p{p}_tick >= p{p}_end_tick - {d['populations'][p]['monitor']['window_steps']} {{",
                    f"    p{p}_event_history_{position}.extend({vector}.iter().map(|&i| (p{p}_tick, i)));", "}"])

    def history(p, event):
        if distributed:
            vector = _v7_event_vector(p, d["populations"][p], event)
            count = d["populations"][p]["count"]
            append([f"{vector}.retain(|&i| mpi.owns(i, {count}, {p}));"])
            pending_events.append((p, event))
        else:
            record_history(p, event)

    def flush_events():
        if not pending_events:
            return
        lines.append(f"        if c{pending_clock}_active {{")
        entries = []
        for p, event in pending_events:
            vector = _v7_event_vector(p, d["populations"][p], event)
            entries.append(f"(&mut {vector}, {d['populations'][p]['count']}, {p})")
        append(["mpi.exchange_spike_batch(&mut [" + ", ".join(entries) + "])?;"])
        for p, event in pending_events:
            record_history(p, event)
        lines.append("        }")
        pending_events.clear()

    def producer(node):
        if node.owner_kind != "population":
            return None
        if node.operation == "event_source":
            return (node.owner_index, "spike")
        if node.operation == "code_object":
            code = d["populations"][node.owner_index]["code_objects"][node.item_index]
            if code["kind"] == "threshold":
                return (node.owner_index, code.get("event_name", "spike"))
        return None

    for node in plan.logical.nodes:
        if distributed:
            event = producer(node)
            # No consumer or other operation may cross an exchange. Distinct
            # consecutive producers on one active clock can share one packet.
            # Bound the combined MPI count, and never alias a mutable vector.
            capacity = sum(d["populations"][p]["count"] for p, _ in pending_events)
            if (event is None or pending_clock != node.clock or event in pending_events
                    or capacity + d["populations"][event[0]]["count"] > 2**31-1):
                flush_events()
            if event is not None:
                pending_clock = node.clock
        lines += [f"        // canonical node {node.id}", f"        if c{node.clock}_active {{"]
        p, item = node.owner_index, node.item_index
        if node.owner_kind == "population":
            pop = d["populations"][p]
            append([f"let time = c{pop['clock']}_tick as f64 * c{pop['clock']}_dt;"])
            symbols = _v7_inputs(model, p, "neuron", "i")
            state_positions = {s["name"]: i for i, s in enumerate(pop["states"])}
            if node.operation == "state_monitor":
                if p not in recorded:
                    recorded.add(p)
                    append([f"if p{p}_tick >= p{p}_end_tick - {pop['monitor']['window_steps']} {{"])
                    for neuron in pop["monitor"]["record"]:
                        values = _v7_inputs(model, p, "neuron", str(neuron))
                        for column, name in enumerate(pop["monitor"]["variables"]):
                            symbol = next(s for s in pop["states"] + pop["parameters"] + pop.get("linked_variables", []) if s["name"] == name)
                            zero = ("false" if symbol["dtype"] == "bool" else
                                    "0.0" if symbol["dtype"] in {"f32", "f64"} else "0")
                            append([f"    p{p}_samples_{column}.push(" + (f"if mpi.owns({neuron}, {pop['count']}, {p}) {{ {_storage_store(symbol, values[name])} }} else {{ {zero} }}" if distributed else _storage_store(symbol, values[name])) + ");"])
                    append(["}"])
            elif node.operation == "spike_monitor":
                append([f"if {'mpi.rank == 0 && ' if distributed else ''}p{p}_tick >= p{p}_end_tick - {pop['monitor']['window_steps']} {{",
                        f"    for &i in &p{p}_fired {{ p{p}_counts[i] += 1; p{p}_spikes.push((p{p}_tick,i)); }}", "}"])
            elif node.operation == "event_monitor":
                monitor = pop["event_monitors"][item]
                vector = _v7_event_vector(p, pop, monitor["event"])
                append([f"for &i in &{vector} {{", f"    p{p}_event_monitor_{item}_events.push((p{p}_tick,i));"])
                for column, name in enumerate(monitor["variables"]):
                    symbol = next(s for s in pop["states"] + pop["parameters"] + pop.get("linked_variables", []) if s["name"] == name)
                    append([f"    p{p}_event_monitor_{item}_samples_{column}.push({_storage_store(symbol, symbols[name])});"])
                append(["}"])
            elif node.operation == "event_source":
                append([f"p{p}_fired.clear();",
                        f"while p{p}_generated_cursor < p{p}_generated_ticks.len() && p{p}_generated_ticks[p{p}_generated_cursor] < p{p}_tick {{ p{p}_generated_cursor += 1; }}",
                        f"while p{p}_generated_cursor < p{p}_generated_ticks.len() && p{p}_generated_ticks[p{p}_generated_cursor] == p{p}_tick {{",
                        f"    p{p}_fired.push(p{p}_generated_indices[p{p}_generated_cursor]); p{p}_generated_cursor += 1;", "}"])
                history(p, "spike")
            elif node.operation == "code_object":
                code = pop["code_objects"][item]
                kind = code["kind"]
                if code.get("adaptive") is not None:
                    if pop["refractory"] is not None and pop["refractory"]["mode"] == "fixed":
                        append([f"for i in 0..{pop['count']} {{",
                                f"    p{p}_not_refractory[i] = u8::from(p{p}_tick >= p{p}_refractory_until[i]);"])
                    else:
                        append([f"for i in 0..{pop['count']} {{"])
                    append(["    " + line for line in adaptive_lines(p, item, pop, code)])
                    append(["}"])
                    lines.append("        }")
                    continue
                scalar, vector, values = _v7_block(
                    model, code, p, "neuron", clock=node.clock,
                    overrides={"dt": f"p{p}_dt"})
                offload = gpu_offload and kind == "state_update"
                if offload:
                    from .mpi_gpu import emit_update
                    append(["if mpi_gpu.enabled() {"] + emit_update(model, p, item) + ["} else {"])
                if distributed:
                    append(['mpi_population_step(|| {'])
                append(scalar)
                event = code.get("event_name", "spike")
                event_vector = _v7_event_vector(p, pop, event) if kind in {"threshold", "reset"} else None
                if kind == "threshold":
                    append([f"{event_vector}.clear();"])
                append([f"for &i in &{event_vector} {{" if kind == "reset" else (f"for i in p{p}_start..p{p}_stop {{" if distributed else f"for i in 0..{pop['count']} {{")])
                if distributed and kind == "reset":
                    append([f"if !mpi.owns(i, {pop['count']}, {p}) {{ continue; }}"])
                if kind == "state_update" and pop["refractory"] is not None and pop["refractory"]["mode"] == "fixed":
                    append([f"    p{p}_not_refractory[i] = u8::from(p{p}_tick >= p{p}_refractory_until[i]);"])
                append(vector)
                if kind == "threshold":
                    gate = values["_cond"]
                    if event == "spike" and pop["refractory"] is not None:
                        gate += f" && p{p}_not_refractory[i] != 0"
                    append([f"if {gate} {{", f"    {event_vector}.push(i);"])
                    if event == "spike" and pop["refractory"] is not None:
                        append([f"    p{p}_lastspike[i] = time; p{p}_not_refractory[i] = 0;"])
                        if pop["refractory"]["mode"] == "fixed":
                            append([f"    p{p}_refractory_until[i] = p{p}_tick + p{p}_period_ticks;"])
                    append(["}"])
                else:
                    for name in code["effects"]["writes"]:
                        if name == "not_refractory":
                            append([f"p{p}_not_refractory[i] = u8::from({values[name]});"])
                        else:
                            index = state_positions[name]
                            append([f"p{p}_state_{index}[i] = {_storage_store(pop['states'][index], values[name])};"])
                append(["}"])
                if distributed:
                    append(['});'])
                if offload:
                    append(["}"])
                if kind == "threshold":
                    history(p, event)
            else:
                raise ValueError(f"unsupported canonical node {node.operation}")
        else:
            q = p
            synapse = d["synapses"][q]
            instance = model["instance"]["synapses"][q]
            if node.operation == "state_monitor":
                if distributed:
                    raise ValueError(
                        "Synapses StateMonitor is not supported by distributed AOT")
                monitor = synapse["state_monitors"][item]
                need_source = any(source["kind"] == "pre_state"
                                  for source in monitor["sources"])
                need_target = any(source["kind"] == "post_state"
                                  for source in monitor["sources"])
                synapse_states = {
                    state["name"]: position
                    for position, state in enumerate(synapse["states"])}
                source_states = {
                    state["name"]: position
                    for position, state in enumerate(
                        d["populations"][synapse["source_population"]]["states"])}
                target_states = {
                    state["name"]: position
                    for position, state in enumerate(
                        d["populations"][synapse["target_population"]]["states"])}
                record = monitor["record"]
                topology = instance.get("topology", {"kind": "explicit"})
                edge_count = (len(instance["source"])
                              if topology["kind"] == "explicit"
                              else topology["edge_count"])
                if record:
                    if (len(record) == edge_count and
                            all(edge == position
                                for position, edge in enumerate(record))):
                        append([f"for edge in 0..s{q}_edge_count {{"])
                    else:
                        indices = ",".join(f"{edge}usize" for edge in record)
                        append([f"for &edge in &[{indices}] {{"])
                    append(_endpoints(
                        model, q, need_source=need_source,
                        need_target=need_target))
                    for column, source in enumerate(monitor["sources"]):
                        if source["kind"] == "synapse_state":
                            value = (
                                f"s{q}_state_{synapse_states[source['name']]}[edge]")
                        elif source["kind"] == "pre_state":
                            population = synapse["source_population"]
                            value = (
                                f"p{population}_state_{source_states[source['name']]}"
                                "[source_state]")
                        elif source["kind"] == "post_state":
                            population = synapse["target_population"]
                            value = (
                                f"p{population}_state_{target_states[source['name']]}"
                                "[target_state]")
                        elif source["kind"] == "linked":
                            position = next(
                                position for position, linked in enumerate(
                                    synapse.get("linked_variables", []))
                                if linked["name"] == source["name"])
                            value = _v7_synapse_link_value(model, q, position)
                        else:
                            raise ValueError("invalid Synapses StateMonitor source")
                        append([
                            f"s{q}_monitor_{item}_samples_{column}.push({value});"])
                    append(["}"])
                lines.append("        }")
                continue
            code = synapse["code_objects"][item]
            kind = code["kind"]
            if distributed:
                from .mpi_additive import shared_additive
                shared = shared_additive(model, q, code)
                if shared is not None:
                    append(shared)
                    lines.append("        }")
                    continue
            source_pop, target_pop = synapse["source_population"], synapse["target_population"]
            # Brian's synaptic runner uses its own activation clock while
            # the dt variable still belongs to the source Synapses clock.
            overrides = ({"dt": f"p{source_pop}_dt"}
                         if kind in {"synapse_run_regularly", "summed_variable"} else None)
            time_clock = (d["populations"][source_pop]["clock"]
                          if overrides is not None else node.clock)
            append([f"let time = c{time_clock}_tick as f64 * c{time_clock}_dt;"])
            scalar, vector, symbols = _v7_block(
                model, code, source_pop, "synapse", "edge", q,
                clock=node.clock, overrides=overrides,
                summed_caches=summed_caches.get(f"{q}/{item}", ()))
            append(scalar)
            if kind in {"synapses", "synapses_post"}:
                r, pathway = next((r, path) for r, path in enumerate(instance["pathways"]) if path["name"] == code["pathway_name"])
                prefix = f"s{q}_" if r == 0 else f"s{q}p{r}_"
                queue = f"plan_s{q}p{r}_queue"
                endpoint = "source" if pathway["kind"] == "pre" else "target"
                p = synapse[f"{endpoint}_population"]
                begin, count = synapse[f"{endpoint}_start"], synapse[f"{endpoint}_count"]
                events = _v7_event_vector(p, d["populations"][p], pathway["event"])
                delays = pathway["delay_ticks"]
                uniform = bool(delays) and all(delay == delays[0] for delay in delays)
                delay = f"{prefix}delay_ticks" if uniform else f"{prefix}delay_ticks[edge]"
                source_batch = distributed and uniform and endpoint == "source"
                append([f"for &neuron in &{events} {{", *([f"    if s{q}_local_edge_count == 0 {{ break; }}"] if distributed else []), f"    if neuron < {begin} || neuron >= {begin+count} {{ continue; }}",
                        f"    let endpoint = neuron - {begin};"])
                if source_batch:
                    append([f"    let delivery = p{p}_tick + {delay};",
                            f"    if delivery < p{p}_end_tick && s{q}_offsets[endpoint] != s{q}_offsets[endpoint+1] {{ let slot = delivery % {queue}.len(); {queue}[slot].push(endpoint); }}"])
                else:
                    append(_edges_for_endpoint(model, q, endpoint, "endpoint", indent="    ", distributed=distributed))
                    append([f"        let delivery = p{p}_tick + {delay};",
                            f"        if delivery < p{p}_end_tick {{ let slot = delivery % {queue}.len(); {queue}[slot].push(edge); }}", "    }"])
                append(["}", f"let slot = p{p}_tick % {queue}.len();",
                        f"let mut active_edges = std::mem::take(&mut {queue}[slot]);"])
                if source_batch:
                    append(["for &source in &active_edges {"])
                    append(_edges_for_endpoint(model, q, "source", "source", distributed=True))
                    append([f"let target = s{q}_target_index[edge] as usize;",
                            f"let source_state = source + {synapse['source_start']};",
                            f"let target_state = target + {synapse['target_start']};"])
                else:
                    append(["for &edge in &active_edges {"])
                    need_source, need_target = _endpoint_requirements(synapse, code)
                    append(_endpoints(model, q, distributed=distributed,
                                      need_source=need_source, need_target=need_target))
                if distributed:
                    # Storage is compact; counter RNG still uses the original edge ID.
                    vector = [line.replace("edge as u64", f"s{q}_original_edges[edge] as u64") for line in vector]
                    # Next Brain's bounded-candidate contract: active_after is
                    # birth time plus the pre delay. A previous generation's
                    # delayed emission must never update a newly born edge,
                    # including its automatically lowered event-driven traces.
                    state_positions = {s['name']: i for i, s in enumerate(synapse['states'])}
                    if {'live', 'born', 'active_after'} <= state_positions.keys():
                        pre_r, pre_path = next((k, path) for k, path in enumerate(instance['pathways']) if path['kind'] == 'pre')
                        pre_prefix = f's{q}_' if pre_r == 0 else f's{q}p{pre_r}_'
                        pre_uniform = len(set(pre_path['delay_ticks'])) == 1
                        pre_delay = f'{pre_prefix}delay_ticks' if pre_uniform else f'{pre_prefix}delay_ticks[edge]'
                        active = f's{q}_state_{state_positions["active_after"]}[edge]'
                        append([f'let birth_tick = (({active} / p{p}_dt).round() as usize).saturating_sub({pre_delay});',
                                f'if p{p}_tick.saturating_sub({delay}) < birth_tick {{ continue; }}'])
                append(vector)
                append(_synapse_writes(model, q, code, symbols))
                counter = "delivered" if pathway["kind"] == "pre" else "post_delivered"
                append([f"s{q}_{counter} += 1;", "}"])
                if source_batch:
                    append(["}"])
                append(["active_edges.clear();", f"{queue}[slot] = active_edges;"])
            elif kind == "summed_variable":
                endpoint = "source" if code["summed_target"] == "pre" else "target"
                p = synapse[f"{endpoint}_population"]
                begin, count = synapse[f"{endpoint}_start"], synapse[f"{endpoint}_count"]
                index, symbol = next((i, s) for i, s in enumerate(d["populations"][p]["states"]) if s["name"] == code["summed_state"])
                zero = "0.0" if symbol["dtype"] in {"f32", "f64"} else "0"
                if distributed:
                    append([f"for target_state in {begin}.max(p{p}_start)..{begin+count}.min(p{p}_stop) {{ p{p}_state_{index}[target_state] = {zero}; }}",
                            f"for edge in 0..s{q}_local_edge_count {{"])
                else:
                    append([f"p{p}_state_{index}[{begin}..{begin+count}].fill({zero});", f"for edge in 0..s{q}_edge_count {{"])
                need_source, need_target = _endpoint_requirements(
                    synapse, code, reduction_target=code["summed_target"])
                append(_endpoints(model, q, distributed=distributed,
                                  need_source=need_source, need_target=need_target))
                append(vector)
                from .native import _storage_load
                location = f"p{p}_state_{index}[{endpoint}_state]"
                value = _storage_store(symbol, f"{_storage_load(symbol, location)} + {symbols['_synaptic_var']}")
                append([f"{location} = {value};", "}"])
            elif kind in {"synapse_state_update", "synapse_subexpression_update", "synapse_run_regularly"}:
                append([f"for edge in 0..s{q}_{'local_edge_count' if distributed else 'edge_count'} {{"])
                need_source, need_target = _endpoint_requirements(synapse, code)
                append(_endpoints(model, q, distributed=distributed,
                                  need_source=need_source, need_target=need_target))
                append(vector)
                append(_synapse_writes(model, q, code, symbols))
                append(["}"])
            else:
                raise ValueError(f"unsupported canonical synapse operation {kind}")
        lines.append("        }")
    flush_events()
    for p, pop in enumerate(d["populations"]):
        lines.append(f"        if p{p}_active {{ p{p}_last_fired.clone_from(&p{p}_fired); p{p}_tick += 1; }}")
    for c in range(len(d["clocks"])):
        lines.append(f"        if c{c}_active {{ c{c}_tick += 1; }}")
    lines.append("        phase_finish(&mut phase_groups_seconds, phase_groups_started);")
    return lines
