"""Run one MPI/build command inside an already-limited Linux cgroup.

The launcher creates a transient systemd service as an unprivileged user.
This child verifies actual limits/mounts, caps file writes, and records evidence.
It does not configure host services or infer available resources from total RAM.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def wait_with_disk_reserve(process, volume, minimum_bytes, timeout):
    deadline = time.monotonic()+timeout
    minimum_observed = None
    while process.poll() is None:
        stats = os.statvfs(volume)
        available = stats.f_bavail*stats.f_frsize
        minimum_observed = available if minimum_observed is None else min(minimum_observed,available)
        if available < minimum_bytes:
            raise RuntimeError('data-volume reserve crossed during execution')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args,timeout)
        try:
            process.wait(timeout=min(5,remaining))
        except subprocess.TimeoutExpired:
            continue
    return minimum_observed


def run(args):
    started = time.monotonic()
    output = args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    cgroup = next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    root = Path('/sys/fs/cgroup')/cgroup.lstrip('/')
    def facts():
        names = ['memory.max','memory.swap.max','memory.current','memory.peak','memory.events','cpu.max','cpu.stat','pids.max','cpuset.cpus.effective']
        return {name:(root/name).read_text().strip() for name in names if (root/name).exists()}
    before = facts()
    report = {'schema':'b2-mpi-resource-guard-v1','command':args.command,'host':os.uname().nodename,
              'uid':os.getuid(),'cgroup':cgroup,'before':before,'admitted':False}
    process = None
    try:
        if os.getuid() == 0:
            raise RuntimeError('simulation/build must run as unprivileged user')
        if not args.volume.is_mount():
            raise RuntimeError('required data mount is unavailable')
        if args.volume.stat().st_dev == Path('/').stat().st_dev and not args.allow_root_volume:
            raise RuntimeError('root-volume storage requires explicit authorization')
        if not output.resolve().is_relative_to(args.volume.resolve()):
            raise RuntimeError('guard report must reside on the data volume')
        available = os.statvfs(args.volume).f_bavail*os.statvfs(args.volume).f_frsize
        if available < args.min_free_gib*2**30:
            raise RuntimeError('insufficient free data-volume space')
        memory = {key:int(value.split()[0])*1024 for key,value in
                  (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                  if key in ('MemAvailable','MemTotal')}
        reserve = min(64*2**30,memory['MemTotal']//4)
        if memory['MemAvailable'] < args.memory_mib*2**20 + reserve:
            raise RuntimeError('insufficient host memory headroom')
        report['host_memory_bytes'] = memory
        report['reserved_host_memory_bytes'] = reserve
        if before['memory.max']=='max' or int(before['memory.max'])>args.memory_mib*2**20 or before['memory.swap.max']!='0':
            raise RuntimeError('required memory/swap cgroup limits are not active')
        quota,period = before['cpu.max'].split()
        if quota=='max' or int(quota)*100>args.cpu_percent*int(period):
            raise RuntimeError('required CPU cgroup quota is not active')
        if before['pids.max']=='max' or int(before['pids.max'])>64:
            raise RuntimeError('required task-count limit is not active')
        report.update(admitted=True,data_free_bytes=available,file_limit_bytes=args.file_mib*2**20,
                      root_volume_allowed=args.allow_root_volume,data_volume=str(args.volume.resolve()),minimum_free_bytes=args.min_free_gib*2**30,disk_check_interval_seconds=5,data_device=args.volume.stat().st_dev,root_device=Path('/').stat().st_dev)
        output.write_text(json.dumps(report,indent=2)+'\n')
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE,(args.file_mib*2**20,args.file_mib*2**20))
        process = subprocess.Popen(args.command,stdin=subprocess.DEVNULL,start_new_session=True,preexec_fn=limits)
        try:
            report['minimum_observed_free_bytes'] = wait_with_disk_reserve(
                process,args.volume,args.min_free_gib*2**30,args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            raise
        report['returncode'] = process.returncode
        return process.returncode if process.returncode>=0 else 128-process.returncode
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        report['after'] = facts()
        report['wall_seconds'] = time.monotonic()-started
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--volume',type=Path,default=Path('/data/brick2'))
    p.add_argument('--allow-root-volume',action='store_true',help='Explicit local-root storage authorization; retain free-space/file-size checks')
    p.add_argument('--memory-mib',type=int,default=2048)
    p.add_argument('--cpu-percent',type=int,default=200)
    p.add_argument('--file-mib',type=int,default=64)
    p.add_argument('--min-free-gib',type=int,default=100)
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.command[:1]==['--']:a.command=a.command[1:]
    if not a.command or min(a.memory_mib,a.cpu_percent,a.file_mib,a.min_free_gib,a.timeout)<=0:
        p.error('positive resource limits and a command are required')
    raise SystemExit(run(a))
