"""Code generation and execution checks for nested Brian array references."""
import re
import shutil
import subprocess

import brian2 as b
import numpy as np
import pytest

from brian2.codegen.codeobject import create_runner_codeobj
from brian2.codegen.generators.cpp_generator import CPPCodeGenerator
from brian2.codegen.generators.cython_generator import CythonCodeGenerator
from brian2.codegen.generators.numpy_generator import NumpyCodeGenerator
from brian2.codegen.runtime.cython_rt.cython_rt import CythonCodeObject
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.codegen.statements import Statement
from brian2.core.variables import ArrayVariable
from brian2.devices.device import get_device


def test_codeobject_collects_transitive_indices(monkeypatch):
    b.set_device('runtime')
    b.start_scope()
    source = b.NeuronGroup(3, 'gain:1\nroute:integer')
    target = b.NeuronGroup(2, 'out:1\npick:integer')
    target.variables.add_reference('route', source, 'route', index='pick')
    target.variables.add_reference('peer', source, 'gain', index='route')
    monkeypatch.setattr(get_device(), 'code_object', lambda **kwargs: kwargs)
    result = create_runner_codeobj(
        target, 'out = peer', 'stateupdate', run_namespace={},
        codeobj_class=CythonCodeObject, check_units=False,
    )
    assert {'out', 'peer', 'route', 'pick'} <= set(result['variables'])
    assert result['variables']['pick'] is target.variables['pick']


def test_codeobject_rejects_index_cycle(monkeypatch):
    b.set_device('runtime')
    b.start_scope()
    target = b.NeuronGroup(2, 'out:1\npick:integer\nroute:integer')
    monkeypatch.setattr(get_device(), 'code_object', lambda **kwargs: kwargs)
    with pytest.raises(ValueError, match='[Cc]yclic.*index'):
        create_runner_codeobj(
            target, 'out = route', 'stateupdate', run_namespace={},
            codeobj_class=CythonCodeObject, check_units=False,
            variable_indices={'route': 'pick', 'pick': 'route'},
        )


def test_codeobject_preserves_template_owned_self_index(monkeypatch):
    b.set_device('runtime')
    b.start_scope()
    target = b.SpikeGeneratorGroup(2, [0], [0]*b.ms)
    monkeypatch.setattr(get_device(), 'code_object', lambda **kwargs: kwargs)
    result = create_runner_codeobj(
        target, '', 'spikegenerator', run_namespace={},
        codeobj_class=CythonCodeObject, check_units=False,
    )
    assert 'spike_number' in result['variables']
    assert result['variable_indices']['spike_number'] == 'spike_number'


def generator(cls, conditional=False, cycle=False):
    b.set_device('runtime')
    variables = {
        name: ArrayVariable(name=name, owner=None, size=4, device=get_device(), dtype=dtype)
        for name, dtype in (
            ('zpick', np.int32), ('aroute', np.int32), ('peer', np.float64),
            ('out', np.float64), ('gate', bool),
        )
    }
    indices = dict(zpick='_idx', aroute='zpick', peer='aroute', out='_idx', gate='aroute')
    if cycle:
        indices['zpick'] = 'aroute'
    if conditional:
        variables['out'].set_conditional_write(variables['gate'])
    return cls(variables, indices, None, {'_idx'}, None, 'nested', 'stateupdate')


@pytest.mark.parametrize('cls', [CythonCodeGenerator, CPPCodeGenerator, NumpyCodeGenerator])
@pytest.mark.parametrize('kind', ['read', 'explicit_root', 'write_root', 'conditional'])
def test_index_reads_follow_dependencies(cls, kind):
    gen = generator(cls, conditional=kind == 'conditional')
    statements = [Statement('out', '=', '1' if kind == 'conditional' else
                            'peer + zpick' if kind == 'explicit_root' else 'peer', '', np.float64)]
    if kind == 'write_root':
        statements.append(Statement('zpick', '+=', '1', '', np.int32))
    read, write, indices, _ = gen.arrays_helper(statements)
    assert {'zpick', 'aroute'} <= indices
    assert not read & indices
    if cls is CythonCodeGenerator:
        lines = gen.translate_to_read_arrays(read, indices)
    elif cls is CPPCodeGenerator:
        lines = gen.translate_to_read_arrays(read, write, indices)
    else:
        lines = gen.read_arrays(read, write, indices, gen.variables, gen.variable_indices)
    names = [re.search(r'(\w+) = ', line).group(1) for line in lines]
    assert len(names) == len(set(names))
    assert names.index('zpick') < names.index('aroute')
    assert names.index('aroute') < names.index('gate' if kind == 'conditional' else 'peer')


@pytest.mark.parametrize('cls', [CythonCodeGenerator, CPPCodeGenerator, NumpyCodeGenerator])
@pytest.mark.parametrize('self_index', [False, True])
def test_cyclic_index_dependencies_fail_before_execution(cls, self_index):
    gen = generator(cls, cycle=True)
    if self_index:
        gen.variable_indices['zpick'] = 'zpick'
    with pytest.raises(ValueError, match='[Cc]yclic.*index'):
        gen.arrays_helper([Statement('out', '=', 'peer', '', np.float64)])


def test_numpy_index_mutation_writes_back_and_keeps_read_snapshot():
    gen = generator(NumpyCodeGenerator)
    statements = [Statement('zpick', '+=', '1', '', np.int32),
                  Statement('out', '=', 'peer', '', np.float64)]
    lines = gen.translate_one_statement_sequence(statements)
    values = dict(zpick=np.array([0, 1, 1, 2], np.int32),
                  aroute=np.array([2, 0, 1, 3], np.int32),
                  peer=np.array([.2, .4, .8, 1.6]), out=np.zeros(4))
    expected = values['peer'][values['aroute'][values['zpick']]].copy()
    namespace = {gen.get_array_name(gen.variables[k]): v for k, v in values.items()}
    exec('\n'.join(lines), namespace)
    np.testing.assert_array_equal(values['zpick'], [1, 2, 2, 3])
    np.testing.assert_array_equal(values['out'], expected)


def test_cpp_index_mutation_keeps_read_snapshot(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler required')
    gen = generator(CPPCodeGenerator)
    statements = [Statement('zpick', '+=', '1', '', np.int32),
                  Statement('out', '=', 'peer', '', np.float64)]
    body = '\n'.join(gen.translate_one_statement_sequence(statements))
    declarations = []
    values = dict(zpick='{0, 1, 1, 2}', aroute='{2, 0, 1, 3}',
                  peer='{.2, .4, .8, 1.6}', out='{}')
    for name, value in values.items():
        var = gen.variables[name]
        declarations.append(f'{gen.c_data_type(var.dtype)} {gen.get_array_name(var)}[4] = {value};')
    out = gen.get_array_name(gen.variables['out'])
    pick = gen.get_array_name(gen.variables['zpick'])
    source = '#include <cstdint>\nint main() {\n' + '\n'.join(declarations)
    source += '\nfor (int _idx=0; _idx<4; ++_idx) {\n' + body + '\n}\n'
    source += f'return !({out}[0]==.8 && {out}[1]==.2 && {out}[2]==.2 && {out}[3]==.4 '
    source += f'&& {pick}[0]==1 && {pick}[1]==2 && {pick}[2]==2 && {pick}[3]==3);\n}}'
    path = tmp_path/'nested.cpp'
    path.write_text(source)
    executable = tmp_path/'nested'
    subprocess.run([compiler, '-std=c++11', str(path), '-o', str(executable)],
                   check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("codeobj_class", [NumpyCodeObject, CythonCodeObject])
def test_nested_reference_executes_through_runtime(codeobj_class):
    b.set_device("runtime")
    b.start_scope()
    source = b.NeuronGroup(3, "gain : 1\nroute : integer", name="nested_source")
    destination = b.NeuronGroup(2, "out : 1\npick : integer", name="nested_destination")
    destination.variables.add_reference("route", source, "route", index="pick")
    destination.variables.add_reference("peer", source, "gain", index="route")
    source.gain = [10, 20, 30]
    source.route = [2, 0, 1]
    destination.pick = [0, 1]
    update = destination.run_regularly(
        "out = peer", dt=0.1 * b.ms, codeobj_class=codeobj_class
    )
    b.Network(source, destination).run(0.1 * b.ms)
    np.testing.assert_array_equal(destination.out[:], [30, 10])
    assert isinstance(update.codeobj, codeobj_class)
