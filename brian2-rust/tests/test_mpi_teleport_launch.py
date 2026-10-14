"""Hydra launch output is checked before it becomes a Teleport command."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('mpi_teleport_launch',Path(__file__).resolve().parents[1]/'tools/mpi_teleport_launch.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PROXY='/test/deps/usr/bin/hydra_pmi_proxy'
LINE=f'HYDRA_LAUNCH: {PROXY} --control-port 192.168.20.23:32123 --rmk user --launcher manual --demux poll --iface bond0 --pgid 0 --retries 10 --usize -2 --pmi-port 0 --gpus-per-proc -2 --gpu-subdevs-per-proc -2 --proxy-id 1'


def test_expected_proxy_is_bound_to_node_and_controller():
    index,args=module.parse_proxy(LINE,PROXY,'192.168.20.23',2)
    assert index==1 and args[0]==PROXY


@pytest.mark.parametrize('bad',[
    LINE.replace(PROXY,'/bin/sh'),
    LINE.replace('192.168.20.23','192.168.20.99'),
    LINE.replace('32123','0'),
    LINE.replace('--proxy-id 1','--proxy-id 2'),
    LINE.replace('--launcher manual','--launcher ssh'),
    LINE+' --proxy-id 0',
    LINE+' --unknown value',
    LINE+' ; touch /tmp/forged',
    LINE.replace(' --control-port 192.168.20.23:32123', ''),
    LINE.replace(' --proxy-id 1', ''),
    LINE.replace(' --launcher manual', ''),
])
def test_malformed_or_redirected_proxy_is_rejected(bad):
    with pytest.raises(ValueError):
        module.parse_proxy(bad,PROXY,'192.168.20.23',2)


@pytest.mark.parametrize('volume,base,guard,valid', [
    ('/data/brick2','/data/brick2/test','/data/brick2/guard.py',True),
    ('/mnt/data','/mnt/data/mam','/mnt/data/mam/guard.py',True),
    ('/','/tmp/test','/tmp/guard.py',False),
    ('/mnt/data','/tmp/test','/mnt/data/guard.py',False),
    ('/mnt/data','/mnt/data/mam','/tmp/guard.py',False),
    ('/mnt/data','/mnt/data/../root','/mnt/data/guard.py',False),
    ('mnt/data','mnt/data/mam','mnt/data/guard.py',False),
    ('/mnt/data','/mnt/data','/mnt/data/guard.py',False),
])
def test_guard_paths_stay_inside_an_explicit_nonroot_volume(volume,base,guard,valid):
    assert module.guard_paths_valid(volume,base,guard) is valid


def test_explicit_root_storage_keeps_path_validation():
    assert module.guard_paths_valid('/', '/atlas-home/0003/mam', '/atlas-home/0003/mam/guard.py', allow_root=True)
    assert not module.guard_paths_valid('/', '/atlas-home/0003/mam', '/atlas-home/0003/mam/guard.py')
    assert not module.guard_paths_valid('/', '/atlas-home/0003/../root', '/atlas-home/0003/mam/guard.py', allow_root=True)
    assert not module.guard_paths_valid('/', '/', '/atlas-home/0003/mam/guard.py', allow_root=True)


def test_runtime_environment_reaches_guarded_controller(tmp_path, monkeypatch):
    class CapturedCommand(Exception):
        pass
    def capture(command, **kwargs):
        args = __import__('shlex').split(command[-1])
        assert 'ASAN_OPTIONS=detect_leaks=0:abort_on_error=1' in args
        assert 'UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1' in args
        raise CapturedCommand
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    with pytest.raises(CapturedCommand):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=1,
                      remote_base='/atlas-home/0003/mam', application=['/bin/true'],
                      output=tmp_path/'run', login='root',
                      guard_script='/atlas-home/0003/mam/guard.py', guard_volume='/',
                      guard_allow_root_volume=True,
                      runtime_environment={
                          'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1',
                          'ASAN_OPTIONS': 'detect_leaks=0:abort_on_error=1'})


@pytest.mark.parametrize('environment', [
    [], {'BAD-NAME': 'x'}, {'1BAD': 'x'}, {'GOOD': 1}, {'GOOD': 'bad\0value'}])
def test_invalid_runtime_environment_never_launches(tmp_path, monkeypatch, environment):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid runtime environment reached subprocess')
    monkeypatch.setattr(module.subprocess, 'Popen', unexpected)
    with pytest.raises(ValueError, match='runtime environment'):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=1,
                      remote_base='/atlas-home/0003/mam', application=['/bin/true'],
                      output=tmp_path/'run', runtime_environment=environment)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('file_options,expected_file_mib', [({},64),({'guard_file_mib':512},512),
    ({'guard_file_mib':12288},12288),({'guard_file_mib':16384},16384),({'guard_file_mib':24576},24576),({'guard_file_mib':131072},131072)])
def test_larger_explicit_budget_reaches_both_cgroup_and_remote_admission(tmp_path, monkeypatch,file_options,expected_file_mib):
    class CapturedCommand(Exception):
        pass
    def capture(command, **kwargs):
        args = __import__('shlex').split(command[-1])
        assert '--pipe' not in args
        assert '--wait' in args and '--collect' in args
        assert '--property=StandardOutput=journal' in args
        assert '--property=StandardError=journal' in args
        assert '--property=MemoryMax=32768M' in args
        assert args[args.index('--memory-mib') + 1] == '32768'
        assert args[args.index('--file-mib') + 1] == str(expected_file_mib)
        assert '--property=MemorySwapMax=0' in args
        assert '--uid=rock' in args
        raise CapturedCommand
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    with pytest.raises(CapturedCommand):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=1,
                      remote_base='/atlas-home/0003/mam', application=['/atlas-home/0003/mam/run'],
                      output=tmp_path/'run', login='root', guard_script='/atlas-home/0003/mam/guard.py',
                      guard_volume='/', guard_allow_root_volume=True, guard_memory_mib=32768,**file_options)


@pytest.mark.parametrize('budget', [True, 63, 262145, 32768.5])
def test_invalid_memory_budget_is_rejected_before_launch(tmp_path, budget):
    with pytest.raises(ValueError, match='bounded resources'):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=1,
                      remote_base='/data/brick2/mam', application=['/test/run'],
                      output=tmp_path/'run', login='root', guard_script='/data/brick2/guard.py',
                      guard_memory_mib=budget)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('budget',[True,0,131073,64.5])
def test_invalid_file_budget_is_rejected_before_launch(tmp_path,budget):
    with pytest.raises(ValueError,match='bounded resources'):
        module.launch(nodes=['node'],ips=['192.168.0.1'],ranks_per_node=1,
                      remote_base='/data/brick2/mam',application=['/test/run'],
                      output=tmp_path/'run',login='root',guard_script='/data/brick2/guard.py',
                      guard_file_mib=budget)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('count,quota', [(1,100), (8,800), (32,3200), (40,4000), (64,6400)])
def test_explicit_cpu_width_and_quota_reach_remote_cgroup(tmp_path, monkeypatch, count, quota):
    class CapturedCommand(Exception):
        pass
    def capture(command, **kwargs):
        args = __import__('shlex').split(command[-1])
        assert f'--property=AllowedCPUs=0-{count-1}' in args
        assert f'--property=CPUQuota={quota}%' in args
        assert args[args.index('--cpu-percent')+1] == str(quota)
        assert '--property=MemorySwapMax=0' in args
        assert '--property=TasksMax=64' in args
        assert '--property=KillMode=control-group' in args
        raise CapturedCommand
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    with pytest.raises(CapturedCommand):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=8,
                      remote_base='/atlas-home/0003/mam', application=['/atlas-home/0003/mam/run'],
                      output=tmp_path/'run', login='root', guard_script='/atlas-home/0003/mam/guard.py',
                      guard_volume='/', guard_allow_root_volume=True,
                      guard_cpu_count=count, guard_cpu_percent=quota)


@pytest.mark.parametrize('options', [dict(guard_cpu_count=True), dict(guard_cpu_count=0),
    dict(guard_cpu_count=65), dict(guard_cpu_count=8.5), dict(guard_cpu_percent=801),
    dict(guard_cpu_count=64,guard_cpu_percent=6401)])
def test_cpu_allocation_rejected_before_remote_launch(tmp_path, monkeypatch, options):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid resource request reached subprocess')
    monkeypatch.setattr(module.subprocess, 'Popen', unexpected)
    with pytest.raises(ValueError, match='bounded resources'):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=8,
                      remote_base='/atlas-home/0003/mam', application=['/atlas-home/0003/mam/run'],
                      output=tmp_path/'run', login='root', guard_script='/atlas-home/0003/mam/guard.py',
                      guard_volume='/', guard_allow_root_volume=True, **options)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('cpus', [[0, 12, 24, 36, 48, 60, 72, 84], (84, 72, 60, 48, 36, 24, 12, 0)])
def test_explicit_cpu_ids_preserve_quota_and_cpu_count(tmp_path, monkeypatch, cpus):
    class CapturedCommand(Exception):
        pass
    def capture(command, **kwargs):
        args = __import__('shlex').split(command[-1])
        assert '--property=AllowedCPUs=0,12,24,36,48,60,72,84' in args
        assert '--property=CPUQuota=800%' in args
        assert args[args.index('--cpu-percent')+1] == '800'
        assert '--property=MemorySwapMax=0' in args
        raise CapturedCommand
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    with pytest.raises(CapturedCommand):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=8,
                      remote_base='/atlas-home/0003/mam', application=['/atlas-home/0003/mam/run'],
                      output=tmp_path/'run', login='root', guard_script='/atlas-home/0003/mam/guard.py',
                      guard_volume='/', guard_allow_root_volume=True,
                      guard_cpu_count=8, guard_cpu_percent=800, guard_cpu_ids=cpus)


@pytest.mark.parametrize('cpus', [[], [0], [0]*8, [0,1,2,3,4,5,6,True],
    [0,1,2,3,4,5,6,-1], [0,1,2,3,4,5,6,4096], '0-7', set(range(8))])
def test_invalid_explicit_cpu_ids_never_start_remote_job(tmp_path, monkeypatch, cpus):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid CPU set reached subprocess')
    monkeypatch.setattr(module.subprocess, 'Popen', unexpected)
    with pytest.raises(ValueError, match='bounded resources'):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=8,
                      remote_base='/atlas-home/0003/mam', application=['/atlas-home/0003/mam/run'],
                      output=tmp_path/'run', login='root', guard_script='/atlas-home/0003/mam/guard.py',
                      guard_volume='/', guard_allow_root_volume=True,
                      guard_cpu_count=8, guard_cpu_percent=800, guard_cpu_ids=cpus)
    assert not (tmp_path/'run').exists()


def test_explicit_cpu_ids_cannot_be_silently_ignored_without_guard(tmp_path):
    with pytest.raises(ValueError, match='require a guard'):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=8,
                      remote_base='/atlas-home/0003/mam', application=['/atlas-home/0003/mam/run'],
                      output=tmp_path/'run', guard_cpu_ids=list(range(8)))
    assert not (tmp_path/'run').exists()


def test_explicit_teleport_proxy_reaches_remote_command(tmp_path, monkeypatch):
    class CapturedCommand(Exception):
        pass
    def capture(command, **kwargs):
        assert command[:4] == ['tsh', '--proxy=example.test:443', 'ssh', 'root@node']
        raise CapturedCommand
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    with pytest.raises(CapturedCommand):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=1,
                      remote_base='/atlas-home/0003/mam', application=['/test/run'],
                      output=tmp_path/'run', login='root',
                      tsh_args=['--proxy=example.test:443'])


@pytest.mark.parametrize('bad', [None, 'proxy', [''], [1]])
def test_invalid_teleport_arguments_do_not_launch(tmp_path, monkeypatch, bad):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid Teleport arguments reached subprocess')
    monkeypatch.setattr(module.subprocess, 'Popen', unexpected)
    with pytest.raises(ValueError, match='Teleport arguments'):
        module.launch(nodes=['node'], ips=['192.168.0.1'], ranks_per_node=1,
                      remote_base='/atlas-home/0003/mam', application=['/test/run'],
                      output=tmp_path/'run', tsh_args=bad)
    assert not (tmp_path/'run').exists()


def test_primary_volume_and_reserve_are_bound_to_controller(tmp_path,monkeypatch):
    class Captured(Exception):pass
    def capture(command,**kwargs):
        args=__import__('shlex').split(command[-1])
        assert args[args.index('--volume')+1]=='/data/brick2'
        assert args[args.index('--output')+1].startswith('/data/brick2/mam/guards/')
        assert '/data/brick2/mam/guard.py' in args
        assert args[args.index('--min-free-gib')+1]=='128'
        raise Captured
    monkeypatch.setattr(module.subprocess,'Popen',capture)
    with pytest.raises(Captured):
        module.launch(nodes=['node23','node81'],ips=['192.168.20.23','192.168.30.81'],ranks_per_node=8,
            remote_base='/atlas-home/0003/mam',application=['/atlas-home/0003/mam/run'],output=tmp_path/'run',
            login='root',guard_script='/atlas-home/0003/mam/guard.py',guard_volume='/',guard_allow_root_volume=True,
            guard_min_free_gib=128,guard_file_mib=131072,
            guard_node_overrides={'node23':dict(volume='/data/brick2',remote_base='/data/brick2/mam',script='/data/brick2/mam/guard.py')})


@pytest.mark.parametrize('options',[
    dict(guard_min_free_gib=True),dict(guard_min_free_gib=0),dict(guard_min_free_gib=4097),
    dict(guard_node_overrides={'unknown':{}}),dict(guard_node_overrides={'node':{}}),
    dict(guard_node_overrides={'node':dict(volume='/data',remote_base='/home/rock',script='/data/guard.py')})])
def test_invalid_primary_guard_configuration_does_not_launch(tmp_path,options):
    with pytest.raises(ValueError):
        module.launch(nodes=['node'],ips=['192.168.0.1'],ranks_per_node=1,remote_base='/atlas-home/0003/mam',
            application=['/atlas-home/0003/mam/run'],output=tmp_path/'run',login='root',guard_script='/atlas-home/0003/mam/guard.py',
            guard_volume='/',guard_allow_root_volume=True,**options)
    assert not (tmp_path/'run').exists()
