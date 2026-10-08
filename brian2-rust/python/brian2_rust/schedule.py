"""Canonical whole-model schedule and effect graph for stable B2IR v1."""


DEFAULT_BASE_SLOTS = (
    "start", "groups", "thresholds", "synapses", "resets", "end",
)


def pathway_sample_lags(definition):
    """Endpoint-clock lag between threshold emission and pathway sampling.

    A pathway ahead of its event threshold consumes the previous tick's flags.
    This is a scheduling property, shared by reference, AOT and GPU lifecycles.
    Missing threshold producers leave empty flags and need no reconstruction.
    """
    thresholds, pathways = {}, []
    for ordinal, node in enumerate(definition['schedule']['nodes']):
        owner = node['owner_index']
        if node['owner_kind'] == 'population':
            if node['operation'] == 'event_source':
                thresholds[owner, 'spike'] = ordinal
            elif node['operation'] == 'code_object':
                code = definition['populations'][owner]['code_objects'][node['item_index']]
                if code['kind'] == 'threshold':
                    thresholds[owner, code['event_name']] = ordinal
        elif (node['owner_kind'] == 'synapse' and
              node['operation'] == 'code_object'):
            syn = definition['synapses'][owner]
            code = syn['code_objects'][node['item_index']]
            if code['kind'] in {'synapses', 'synapses_post'}:
                side = 'source' if code['kind'] == 'synapses' else 'target'
                pathways.append((owner, code, syn[side+'_population'], ordinal))
    return {(owner, code['pathway_name']): int(thresholds.get((population, code['event_name']), -1) > ordinal)
            for owner, code, population, ordinal in pathways}


def expanded_slots(base_slots):
    """Return Brian's canonical before/base/after ordering for each slot."""
    return tuple(
        expanded
        for slot in base_slots
        for expanded in (f"before_{slot}", slot, f"after_{slot}")
    )


def _population_resource(population, category, name):
    return f"population/{population}/{category}/{name}"


def _synapse_resource(synapse, category, name):
    return f"synapse/{synapse}/{category}/{name}"


def _event_resource(population, name="spike"):
    return _population_resource(population, "event", name)


def execution_effects(definition, node):
    """Complete frozen v1 wire effects with the operation's implicit accesses.

    V1's serialized graph predates explicit refractory side effects. Preserve
    its bytes/hashes, but never use that graph alone to prove an optimization.
    This completion is idempotent and does not mutate the hashed Definition.
    """
    reads = set(node["effects"]["reads"])
    writes = set(node["effects"]["writes"])
    if (node.get("operation") == "code_object" and
            node.get("owner_kind") == "population"):
        p = node["owner_index"]
        population = definition["populations"][p]
        refractory = population["refractory"]
        code = population["code_objects"][node["item_index"]]
        if refractory is not None:
            lastspike = _population_resource(p, "refractory", "lastspike")
            available = _population_resource(p, "refractory", "not_refractory")
            if code["kind"] == "threshold" and code["event_name"] == "spike":
                reads.add(available)
                writes.update((lastspike, available))
            elif code["kind"] == "state_update" and refractory["mode"] == "fixed":
                reads.add(lastspike)
                writes.add(available)
        if code["kind"] == "spatial_state_update":
            spatial = population["spatial"]
            states = {item["name"] for item in population["states"]}
            for name in (spatial["voltage"], spatial["capacitance"],
                         spatial["resistivity"], spatial["area"],
                         spatial["r_length_1"], spatial["r_length_2"]):
                category = "state" if name in states else "parameter"
                reads.add(_population_resource(p, category, name))
            writes.update((
                _population_resource(p, "state", spatial["voltage"]),
                _population_resource(p, "state", spatial["membrane_current"]),
            ))
    return {"reads": sorted(reads), "writes": sorted(writes)}


def _population_effects(population_index, population, code):
    states = {item["name"] for item in population["states"]}
    parameters = {item["name"] for item in population["parameters"]}
    linked = {item["name"]: item for item in population.get("linked_variables", [])}

    def resource(name):
        if name in states:
            return _population_resource(population_index, "state", name)
        if name in parameters:
            return _population_resource(population_index, "parameter", name)
        if name in linked:
            item = linked[name]
            return _population_resource(
                item["source_population"], "state", item["source_state"])
        if name in {"lastspike", "not_refractory"}:
            return _population_resource(population_index, "refractory", name)
        return None

    reads = {resource(name) for name in code["effects"]["reads"]}
    for name in code["effects"]["reads"]:
        item = linked.get(name)
        if item is not None and item["index"]["kind"] in {"state", "parameter"}:
            reads.add(_population_resource(
                population_index, item["index"]["kind"],
                item["index"]["name"]))
    writes = {resource(name) for name in code["effects"]["writes"]}
    if code["kind"] == "threshold":
        writes.add(_event_resource(population_index, code["event_name"]))
    elif code["kind"] == "reset":
        reads.add(_event_resource(population_index, code["event_name"]))
    elif code["kind"] == "spatial_state_update":
        spatial = population["spatial"]
        for name in (spatial["voltage"], spatial["capacitance"],
                     spatial["resistivity"], spatial["area"],
                     spatial["r_length_1"], spatial["r_length_2"]):
            reads.add(resource(name))
        writes.update((resource(spatial["voltage"]),
                       resource(spatial["membrane_current"])))
    return {
        "reads": sorted(item for item in reads if item is not None),
        "writes": sorted(item for item in writes if item is not None),
    }


def _synapse_effects(synapse_index, synapse, code):
    source = synapse["source_population"]
    target = synapse["target_population"]
    states = {item["name"] for item in synapse["states"]}
    parameters = {item["name"] for item in synapse["parameters"]}

    def resource(name):
        if name in states:
            return _synapse_resource(synapse_index, "state", name)
        if name in parameters:
            return _synapse_resource(synapse_index, "parameter", name)
        if name in synapse["pre_state_aliases"]:
            state = synapse["pre_state_aliases"][name]
            source_synapse = synapse.get("source_synapse")
            if source_synapse is not None:
                return _synapse_resource(source_synapse, "state", state)
            return _population_resource(
                source, "state", state)
        if name in synapse["post_state_aliases"]:
            state = synapse["post_state_aliases"][name]
            target_synapse = synapse.get("target_synapse")
            if target_synapse is not None:
                return _synapse_resource(target_synapse, "state", state)
            return _population_resource(target, "state", state)
        linked = next((item for item in synapse.get("linked_variables", [])
                       if item["name"] == name), None)
        if linked is not None:
            return _population_resource(
                linked["source_population"], "state", linked["source_state"])
        if name == "not_refractory_post":
            return _population_resource(target, "refractory", "not_refractory")
        return None

    reads = {resource(name) for name in code["effects"]["reads"]}
    writes = {resource(name) for name in code["effects"]["writes"]}
    if code["kind"] == "summed_variable":
        is_pre = code["summed_target"] == "pre"
        endpoint_synapse = synapse.get(
            "source_synapse" if is_pre else "target_synapse")
        if endpoint_synapse is not None:
            writes.add(_synapse_resource(
                endpoint_synapse, "state", code["summed_state"]))
        else:
            endpoint = source if is_pre else target
            writes.add(_population_resource(
                endpoint, "state", code["summed_state"]))
    elif code["kind"] in {"synapses", "synapses_post"}:
        endpoint = source if code["kind"] == "synapses" else target
        reads.add(_event_resource(endpoint, code["event_name"]))
    return {
        "reads": sorted(item for item in reads if item is not None),
        "writes": sorted(item for item in writes if item is not None),
    }


def _node(identifier, operation, owner_kind, owner_index, item_index,
          clock, when, order, name, effects):
    return {
        "id": identifier,
        "operation": operation,
        "owner_kind": owner_kind,
        "owner_index": owner_index,
        "item_index": item_index,
        "clock": clock,
        "when": when,
        "order": order,
        "name": name,
        "effects": effects,
        "dependencies": [],
    }


def build_schedule(definition, instance, network_slots):
    """Build one total Brian order plus conservative resource dependencies."""
    base_slots = list(network_slots)
    if (not base_slots or len(set(base_slots)) != len(base_slots) or
            any(not isinstance(slot, str) or not slot or
                slot.startswith("before_") or slot.startswith("after_")
                for slot in base_slots)):
        raise ValueError("invalid Brian2 base Network schedule")
    slots = list(expanded_slots(base_slots))
    nodes = []
    populations = definition["populations"]
    for p, population in enumerate(populations):
        for c, code in enumerate(population["code_objects"]):
            nodes.append(_node(
                f"population/{p}/code/{c}", "code_object", "population",
                p, c, code["clock"], code["when"], code["order"], code["name"],
                _population_effects(p, population, code)))
        for m, monitor in enumerate(population["state_monitors"]):
            states = {item["name"] for item in population["states"]}
            links = {item["name"]: item
                     for item in population.get("linked_variables", [])}
            reads = []
            for name in monitor["variables"]:
                if name in links:
                    link = links[name]
                    reads.append(_population_resource(
                        link["source_population"], "state", link["source_state"]))
                    if link["index"]["kind"] in {"state", "parameter"}:
                        reads.append(_population_resource(
                            p, link["index"]["kind"], link["index"]["name"]))
                else:
                    reads.append(_population_resource(
                        p, "state" if name in states else "parameter", name))
            reads = sorted(set(reads))
            nodes.append(_node(
                f"population/{p}/state_monitor/{m}", "state_monitor",
                "population", p, m, monitor["clock"],
                monitor.get("when", "start"), monitor.get("order", 0),
                monitor["name"],
                {"reads": reads, "writes": []}))
        for m, monitor in enumerate(population.get("event_monitors", [])):
            state_names = {item["name"] for item in population["states"]}
            links = {item["name"]: item
                     for item in population.get("linked_variables", [])}
            reads = [_event_resource(p, monitor["event"])]
            for name in monitor["variables"]:
                if name in links:
                    link = links[name]
                    reads.append(_population_resource(
                        link["source_population"], "state", link["source_state"]))
                    if link["index"]["kind"] in {"state", "parameter"}:
                        reads.append(_population_resource(
                            p, link["index"]["kind"], link["index"]["name"]))
                else:
                    reads.append(_population_resource(
                        p, "state" if name in state_names else "parameter", name))
            nodes.append(_node(
                f"population/{p}/event_monitor/{m}", "event_monitor",
                "population", p, m, monitor["clock"], monitor["when"],
                monitor["order"], monitor["name"],
                {"reads": sorted(set(reads)), "writes": []}))
        if population["spike_monitor"] is not None:
            nodes.append(_node(
                f"population/{p}/spike_monitor/0", "spike_monitor",
                "population", p, 0, population["clock"], "thresholds", 1,
                population["spike_monitor"],
                {"reads": [_event_resource(p)], "writes": []}))
        if instance["populations"][p]["spike_generator"] is not None:
            nodes.append(_node(
                f"population/{p}/event_source/0", "event_source",
                "population", p, 0, population["clock"], "thresholds", 0,
                population["name"],
                {"reads": [], "writes": [_event_resource(p)]}))

    for q, synapse in enumerate(definition["synapses"]):
        for c, code in enumerate(synapse["code_objects"]):
            nodes.append(_node(
                f"synapse/{q}/code/{c}", "code_object", "synapse",
                q, c, code["clock"], code["when"], code["order"],
                code["name"], _synapse_effects(q, synapse, code)))
        for m, monitor in enumerate(synapse.get("state_monitors", [])):
            reads = []
            for source in monitor["sources"]:
                if source["kind"] == "synapse_state":
                    reads.append(_synapse_resource(q, "state", source["name"]))
                elif source["kind"] == "pre_state":
                    reads.append(_population_resource(
                        synapse["source_population"], "state", source["name"]))
                elif source["kind"] == "post_state":
                    reads.append(_population_resource(
                        synapse["target_population"], "state", source["name"]))
                elif source["kind"] == "linked":
                    linked = next(
                        item for item in synapse.get("linked_variables", [])
                        if item["name"] == source["name"])
                    reads.append(_population_resource(
                        linked["source_population"], "state",
                        linked["source_state"]))
                else:
                    raise ValueError("invalid synapse StateMonitor source")
            nodes.append(_node(
                f"synapse/{q}/state_monitor/{m}", "state_monitor", "synapse",
                q, m, monitor["clock"], monitor.get("when", "start"),
                monitor.get("order", 0), monitor["name"],
                {"reads": sorted(set(reads)), "writes": []}))

    slot_positions = {slot: position for position, slot in enumerate(slots)}
    nodes.sort(key=lambda node: (
        slot_positions[node["when"]], node["order"], node["name"], node["id"]))

    last_writer = {}
    readers = {}
    for node in nodes:
        dependencies = set()
        for resource in node["effects"]["reads"]:
            if resource in last_writer:
                dependencies.add(last_writer[resource])
        for resource in node["effects"]["writes"]:
            if resource in last_writer:
                dependencies.add(last_writer[resource])
            dependencies.update(readers.get(resource, ()))
        node["dependencies"] = sorted(dependencies)
        for resource in node["effects"]["writes"]:
            last_writer[resource] = node["id"]
            readers[resource] = set()
        for resource in node["effects"]["reads"]:
            readers.setdefault(resource, set()).add(node["id"])
    return {"base_slots": base_slots, "slots": slots, "nodes": nodes}
