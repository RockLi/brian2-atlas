"""Launch MPICH Hydra proxies through local Teleport credentials.

Hydra control and MPI data flow over the nodes' private network. The controller
uses Hydra's manual launcher so no credentials or SSH keys are copied to nodes.
Requires the same absolute MPI/application paths on all listed Linux hosts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid


def parse_proxy(line, proxy_path, control_ip, node_count):
    """Accept only the expected Hydra proxy executable and option-value pairs."""
    if not line.startswith("HYDRA_LAUNCH: "):
        raise ValueError("not a Hydra launch line")
    args = shlex.split(line.split(": ", 1)[1])
    if not args or args[0] != proxy_path or len(args) % 2 != 1:
        raise ValueError("unexpected Hydra executable/argument structure")
    permitted = {"--control-port", "--rmk", "--launcher", "--demux", "--iface", "--pgid",
                 "--retries", "--usize", "--pmi-port", "--gpus-per-proc",
                 "--gpu-subdevs-per-proc", "--proxy-id"}
    options = dict(zip(args[1::2], args[2::2], strict=True))
    if len(options) != len(args[1::2]) or not set(options) <= permitted:
        raise ValueError("unknown/duplicate Hydra option")
    if not {"--control-port", "--proxy-id", "--launcher"} <= options.keys():
        raise ValueError("missing required Hydra option")
    address, port = options["--control-port"].rsplit(":", 1)
    if address != control_ip or not 1 <= int(port) <= 65535:
        raise ValueError("unexpected Hydra control address")
    proxy_id = int(options["--proxy-id"])
    if not 0 <= proxy_id < node_count or options.get("--launcher") != "manual":
        raise ValueError("unexpected proxy id or launcher")
    return proxy_id, args


def guard_paths_valid(volume, remote_base, script, *, allow_root=False):
    """Lexical preflight only; the remote guard verifies the actual mount/device."""
    paths = [PurePosixPath(value) for value in (volume, remote_base, script)]
    root, base, guard = paths
    return (all(path.is_absolute() and ".." not in path.parts for path in paths)
            and (allow_root or root != PurePosixPath("/")) and base != root and guard != root
            and base.is_relative_to(root) and guard.is_relative_to(root))


def runtime_environment_valid(environment):
    """Only accept explicit, shell-safe environment variable assignments."""
    return (isinstance(environment, dict)
            and all(isinstance(name, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                    and isinstance(value, str) and "\0" not in value
                    for name, value in environment.items()))


def launch(*, nodes, ips, ranks_per_node, remote_base, application, output,
           interface="bond0", timeout=120, tsh="tsh", tsh_args=(), login="rock", mpi_prefix=None,
           guard_script=None, guard_memory_mib=2048, guard_cpu_percent=200, guard_volume="/data/brick2", guard_allow_root_volume=False,
           guard_file_mib=64, guard_cpu_count=8, guard_cpu_ids=None,
           guard_min_free_gib=None, guard_node_overrides=None, runtime_environment=None):
    if len(nodes) != len(ips) or len(set(nodes)) != len(nodes) or len(set(ips)) != len(ips):
        raise ValueError("nodes and IPs must be distinct paired lists")
    if not nodes or ranks_per_node < 1 or timeout < 1:
        raise ValueError("positive node/rank count and timeout required")
    if not remote_base.startswith("/") or not application:
        raise ValueError("absolute remote base and application required")
    if (not isinstance(tsh_args, (list, tuple))
            or any(not isinstance(value, str) or not value for value in tsh_args)):
        raise ValueError("Teleport arguments must be nonempty strings")
    if guard_script and (login != "root" or not guard_paths_valid(guard_volume, remote_base, guard_script, allow_root=guard_allow_root_volume)
                         or type(guard_memory_mib) is not int or not 64 <= guard_memory_mib <= 262144
                         or type(guard_file_mib) is not int or not 1 <= guard_file_mib <= 131072
                         or type(guard_cpu_count) is not int or not 1 <= guard_cpu_count <= 64
                         or type(guard_cpu_percent) is not int or not 1 <= guard_cpu_percent <= 100*guard_cpu_count):
        raise ValueError("guard requires root-managed user jobs on an explicitly permitted volume with bounded resources")
    if guard_min_free_gib is not None and (not guard_script or type(guard_min_free_gib) is not int
            or not 1 <= guard_min_free_gib <= 4096):
        raise ValueError('minimum disk reserve requires a guard and integer GiB in 1..4096')
    if runtime_environment is not None and not runtime_environment_valid(runtime_environment):
        raise ValueError("runtime environment requires a string mapping with valid variable names")
    if guard_node_overrides is not None:
        if not guard_script or not isinstance(guard_node_overrides,dict) or not set(guard_node_overrides)<=set(nodes):
            raise ValueError('guard node overrides must name configured nodes')
        for override in guard_node_overrides.values():
            if (not isinstance(override,dict) or set(override)!={'volume','remote_base','script'}
                    or any(not isinstance(v,str) for v in override.values())
                    or not guard_paths_valid(override['volume'],override['remote_base'],override['script'],allow_root=guard_allow_root_volume)):
                raise ValueError('guard node overrides must remain on their explicit volume')
    # Default cgroup command and metadata stay unchanged. Explicit IDs permit
    # topology-aware placement without silently increasing quota or CPU count.
    if guard_cpu_ids is not None:
        if (not guard_script or not isinstance(guard_cpu_ids, (list, tuple))
                or len(guard_cpu_ids) != guard_cpu_count
                or any(type(cpu) is not int or not 0 <= cpu < 4096 for cpu in guard_cpu_ids)
                or len(set(guard_cpu_ids)) != guard_cpu_count):
            raise ValueError("explicit CPU IDs require a guard and distinct bounded resources matching CPU count")
        guard_cpu_ids = sorted(guard_cpu_ids)
    guard_cpuset = (",".join(map(str, guard_cpu_ids)) if guard_cpu_ids is not None
                    else "0-"+str(guard_cpu_count-1))
    job_id = "b2mpi-"+uuid.uuid4().hex[:12]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if mpi_prefix is not None and not mpi_prefix.startswith("/"):
        raise ValueError("MPI prefix must be absolute")
    bindir = (mpi_prefix + "/bin") if mpi_prefix else remote_base + "/deps/usr/bin"
    libdir = (mpi_prefix + "/lib") if mpi_prefix else remote_base + "/deps/usr/lib/x86_64-linux-gnu"
    env_args = ["env", "LD_LIBRARY_PATH=" + libdir,
                "UCX_TLS=tcp,self,sm", "UCX_NET_DEVICES="+interface]
    env_args += [f"{name}={value}" for name, value in sorted((runtime_environment or {}).items())]
    # ch3:sock can select an address by global rank; do not depend on DNS.
    env_args += [f"MPICH_INTERFACE_HOSTNAME_R{rank}={ip}"
                 for rank, ip in enumerate(ip for ip in ips for _ in range(ranks_per_node))]
    # A remote timeout bounds proxy/rank lifetime even if Teleport disconnects.
    bounded = ["timeout", "-k", "5", str(timeout)]
    controller_args = env_args + bounded + [bindir + "/mpiexec.hydra", "-launcher", "manual",
        "-iface", interface, "-hosts", ",".join(f"{ip}:{ranks_per_node}" for ip in ips),
        "-n", str(len(nodes)*ranks_per_node), *application]
    processes, commands, transcripts = {}, {}, {}
    observers, observer_commands, control_transcripts = {}, {}, {}
    events = queue.Queue()
    workers = []
    started = time.monotonic()
    error = None

    def spawn(label, node, args):
        unit = job_id+"-"+label
        if guard_script:
            placement = (guard_node_overrides or {}).get(node,{})
            report_base = placement.get('remote_base',remote_base)
            volume = placement.get('volume',guard_volume)
            script = placement.get('script',guard_script)
            # Keep the service's stdio owned by systemd.  A long-lived
            # `systemd-run --pipe` client makes the workload inherit a pipe
            # whose reader is the Teleport session; if that observation
            # session disappears, the next successful diagnostic write can
            # kill mpiexec with SIGPIPE even though every rank is healthy.
            # Journal followers below are observation-only and can disappear
            # without changing the transient service's file descriptors.
            args = ["systemd-run", "--expand-environment=no", "--quiet", "--wait", "--collect",
                    "--service-type=exec", "--unit="+unit, "--uid=rock",
                    "--property=StandardOutput=journal", "--property=StandardError=journal",
                    "--property=MemoryMax="+str(guard_memory_mib)+"M",
                    "--property=MemorySwapMax=0", "--property=CPUQuota="+str(guard_cpu_percent)+"%",
                    "--property=AllowedCPUs="+guard_cpuset, "--property=TasksMax=64",
                    "--property=RuntimeMaxSec="+str(timeout+5), "--property=TimeoutStopSec=5",
                    "--property=KillMode=control-group", "--property=OOMPolicy=continue",
                    "/usr/bin/python3", script, "--output", report_base+"/guards/"+job_id+"-"+label+".json",
                    *(["--allow-root-volume"] if guard_allow_root_volume else []),
                    "--volume", volume, "--memory-mib", str(guard_memory_mib), "--cpu-percent", str(guard_cpu_percent),
                    "--file-mib", str(guard_file_mib),
                    *(["--min-free-gib",str(guard_min_free_gib)] if guard_min_free_gib is not None else []),
                    "--timeout", str(timeout), "--", *args]
        command = [tsh, *tsh_args, "ssh", f"{login}@{node}", shlex.join(args)]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True, bufsize=1)
        processes[label], commands[label], transcripts[label], control_transcripts[label] = process, command, [], []
        def read_control():
            for line in process.stdout:
                events.put((label, "control", line))
            events.put((label, "control", None))
        thread = threading.Thread(target=read_control, daemon=True)
        workers.append(thread)
        thread.start()
        if guard_script:
            journal_args = ["journalctl", "--no-pager", "--output=cat", "--follow",
                            "--lines=all", "--unit="+unit+".service"]
            journal_command = [tsh, *tsh_args, "ssh", f"{login}@{node}", shlex.join(journal_args)]
            observer = subprocess.Popen(journal_command, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, start_new_session=True, bufsize=1)
            observers[label], observer_commands[label] = observer, journal_command
            def read_journal():
                for line in observer.stdout:
                    events.put((label, "journal", line))
                events.put((label, "journal", None))
            journal_thread = threading.Thread(target=read_journal, daemon=True)
            workers.append(journal_thread)
            journal_thread.start()

    spawn("controller", nodes[0], controller_args)
    proxies, closed = set(), set()
    try:
        while True:
            if time.monotonic()-started > timeout+15:
                raise TimeoutError("Teleport/Hydra launch exceeded deadline")
            try:
                label, source, line = events.get(timeout=0.2)
            except queue.Empty:
                if processes["controller"].poll() is not None and "controller" in closed:
                    break
                continue
            if line is None:
                if source == "control":
                    closed.add(label)
                    if label == "controller":
                        break
                continue
            if source == "journal" or not guard_script:
                transcripts[label].append(line)
                with (output / f"{label}.log").open("a") as stream:
                    stream.write(line)
            else:
                control_transcripts[label].append(line)
            if label == "controller" and (source == "journal" or not guard_script) and line.startswith("HYDRA_LAUNCH: "):
                proxy_id, args = parse_proxy(line, bindir+"/hydra_pmi_proxy", ips[0], len(nodes))
                if proxy_id in proxies:
                    raise ValueError("duplicate Hydra proxy launch")
                proxies.add(proxy_id)
                spawn(f"proxy-{proxy_id}", nodes[proxy_id], env_args + bounded + args)
        for process in processes.values():
            process.wait(timeout=max(1, timeout+15-(time.monotonic()-started)))
        if proxies != set(range(len(nodes))) or any(p.returncode for p in processes.values()):
            raise RuntimeError("Hydra/Teleport process failed; see launch transcripts")
    except (Exception, KeyboardInterrupt) as exc:
        error = str(exc)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                try:
                    import os
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for observer in observers.values():
            if observer.poll() is None:
                try:
                    import os
                    os.killpg(observer.pid, signal.SIGTERM)
                    observer.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(observer.pid, signal.SIGKILL)
                    observer.wait()
        for worker in workers:
            worker.join(timeout=1)
        while not events.empty():
            label, source, line = events.get_nowait()
            if line is not None:
                (transcripts if source == "journal" or not guard_script else control_transcripts)[label].append(line)
        for label, lines in transcripts.items():
            (output/f"{label}.log").write_text("".join(lines))
        for label, lines in control_transcripts.items():
            if lines:
                (output/f"{label}-systemd-run.log").write_text("".join(lines))
        result = {"schema": "b2-teleport-hydra-launch-v0", "nodes": nodes, "ips": ips,
            "ranks_per_node": ranks_per_node, "commands": commands,
            "returncodes": {label:p.returncode for label,p in processes.items()},
            "wall_seconds": time.monotonic()-started, "error": error,
            "remote_timeout_seconds": timeout,
            "runtime_environment": runtime_environment,
            "resource_guard": {"script":guard_script,"volume":guard_volume,"allow_root_volume":guard_allow_root_volume,"memory_mib":guard_memory_mib,"cpu_percent":guard_cpu_percent,"cpu_count":guard_cpu_count,**({"cpu_ids":guard_cpu_ids} if guard_cpu_ids is not None else {}),"file_mib":guard_file_mib,"unit_prefix":job_id,**({"minimum_free_gib":guard_min_free_gib} if guard_min_free_gib is not None else {}),**({"node_overrides":guard_node_overrides} if guard_node_overrides is not None else {})} if guard_script else None,
            "transport": "Teleport observation; systemd-owned journal stdio; Hydra/MPI data over private network",
            "observer_commands": observer_commands}
        (output/"launch.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--ips", nargs="+", required=True)
    parser.add_argument("--ranks-per-node", type=int, default=1)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--mpi-prefix", help="MPICH installation prefix (bin/ and lib/); default: extracted Ubuntu packages")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--interface", default="bond0")
    parser.add_argument("--login", default="rock")
    parser.add_argument("--guard-script", help="Resource guard; requires root login, runs workload as rock")
    parser.add_argument("--guard-volume", default="/data/brick2", help="Existing mount, verified remotely; root requires --guard-allow-root-volume")
    parser.add_argument("--guard-allow-root-volume", action="store_true", help="Explicitly allow local root storage; free-space and file-size limits still apply")
    parser.add_argument("--guard-memory-mib", type=int, default=2048,
                        help="Explicit per-service budget (64..262144 MiB); each proxy includes all its local ranks. Remote headroom admission still applies.")
    parser.add_argument("--guard-cpu-percent", type=int, default=200)
    parser.add_argument("--guard-cpu-count", type=int, default=8,
                        help="Explicit CPU affinity width, using CPUs 0..count-1 (1..64). Quota cannot exceed this width; host admission is still required.")
    parser.add_argument("--guard-file-mib", type=int, default=64, help="Explicit per-file write limit (1..24576 MiB); free-space admission still applies")
    parser.add_argument("application", nargs=argparse.REMAINDER)
    args = vars(parser.parse_args())
    if args["application"][:1] == ["--"]:
        args["application"] = args["application"][1:]
    print(json.dumps(launch(**args),indent=2))


if __name__ == "__main__":
    main()
