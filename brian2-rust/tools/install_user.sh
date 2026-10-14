#!/bin/sh
set -eu
# Install one coherent Atlas source snapshot, including both native engines.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
brian_root=$(CDPATH= cd -- "$backend_root/.." && pwd)
install_root=${BRIAN2_ATLAS_HOME:-"$HOME/.local/share/brian2-atlas"}
command_root=${BRIAN2_ATLAS_BIN:-"$HOME/.local/bin"}
python_version=${BRIAN2_ATLAS_PYTHON:-3.14}
command -v uv >/dev/null 2>&1 || { echo 'uv is required to install Atlas' >&2; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo 'cargo/rustup and Rust 1.98.1 are required to build Atlas' >&2; exit 1; }
mkdir -p "$install_root" "$command_root"
install_root=$(CDPATH= cd -- "$install_root" && pwd)
command_root=$(CDPATH= cd -- "$command_root" && pwd)
venv="$install_root/atlas-venv"
uv venv "$venv" --python "$python_version" --allow-existing
uv pip install --python "$venv/bin/python" --reinstall-package brian2-atlas "$brian_root"
# Avoid importing a checkout from the caller's working directory.
cd "$install_root"
"$venv/bin/python" - "$brian_root" "$install_root" "$command_root" <<'PY'
import hashlib,importlib.metadata,json,pathlib,shlex,subprocess,sys
source,install,commands=map(lambda p:pathlib.Path(p).resolve(),sys.argv[1:]);prefix=pathlib.Path(sys.prefix).resolve()
import brian2,brian2_rust
from brian2_rust._runtime import executable_path,source_root
assert source_root() is None
assert pathlib.Path(brian2.__file__).resolve().is_relative_to(prefix)
package=pathlib.Path(brian2_rust.__file__).resolve().parent
assert package.is_relative_to(prefix)
build=json.loads((package/'_bin/build.json').read_text())
launchers={name:pathlib.Path(sys.executable) for name in ['brian2-atlas','brian2-atlas-python']}
launchers.update({name:prefix/'bin'/name for name in ['b2-runner','b2-train']})
for name in ['b2-runner','b2-train']:
 binary=executable_path(name);assert binary.is_relative_to(package/'_bin')
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binaries'][binary.name]
 result=subprocess.run([str(launchers[name]),'--help'],text=True,capture_output=True,timeout=60)
 assert 'usage:' in (result.stdout+result.stderr).lower(),name
for name,executable in launchers.items():
 destination=commands/name
 if destination.exists() and str(install) not in destination.read_text():
  raise RuntimeError(f'Refusing to replace an unrelated command: {destination}; choose BRIAN2_ATLAS_BIN')
 destination.write_text('#!/bin/sh\nexec '+shlex.quote(str(executable))+' "$@"\n');destination.chmod(0o755)
revision=subprocess.run(['git','rev-parse','HEAD'],cwd=source,capture_output=True,text=True)
diff=subprocess.run(['git','diff','HEAD','--'],cwd=source,capture_output=True)
report={'schema':'brian2-atlas-user-install-v2','source':str(source),'source_commit':revision.stdout.strip() if revision.returncode==0 else None,'source_diff_sha256':hashlib.sha256(diff.stdout).hexdigest() if diff.returncode==0 else None,'python':sys.version,'prefix':str(prefix),'distribution':importlib.metadata.version('brian2-atlas'),'brian_compatibility_version':brian2.__version__,'native_build':build,'commands':{name:str(commands/name) for name in launchers}}
(install/'install.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Installed Atlas commands in {commands}; manifest: {install / "install.json"}')
PY
