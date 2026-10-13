# Expanded simulator coverage

This suite varies the model as well as its inputs. Training is not part of it.
See the [compatibility checklist](../COMPATIBILITY.md) for verified behavior,
unsupported features, numeric bounds and the SpiNNaker2 timestamp issue.

The static generator uses seed `20261013`: 20 binary-count networks and 25
signed/biased dense networks. The 20 count networks produce 388 sample rows;
all 45 together produce 752 rows. Every target receives the same source IR
and input array for a given model. SDK workers never receive labels or Atlas
reference outputs. All output directories refuse overwrites.

Use the SDK environments from the [main example](../README.md),
[SpiNNaker2 example](../spinnaker2/README.md) and
[Xylo example](../xylo_temporal/README.md). Verified versions are retained in
the audits; no SDK source was patched.

## Prepare and execute static cases

From the Atlas repository:

```sh
export PYTHONPATH="$PWD:$PWD/brian2-rust/python"
export B2_RUNNER=/absolute/path/to/b2-runner
export RUN="$PWD/output/neuromorphic/backend-coverage"
export CODE="$PWD/brian2-rust/examples/neuromorphic/coverage"
export API="$PWD/brian2-rust/python/brian2_rust"
export SDK_ENV=/absolute/path/to/isolated/linux-sdk-env

python "$CODE/prepare.py" --runner "$B2_RUNNER" --output "$RUN"
```

Akida uses its existing minimal image:

```sh
docker run --rm --network none --cpus 2 --memory 3g \
  -v "$CODE:/code:ro" -v "$API:/api:ro" -v "$RUN:/work" \
  --entrypoint python atlas-akida-minimal:2.19.3 \
  /code/sdk_worker.py --target akida --runtime /api/neuromorphic_runtime.py --run /work
```

Run Lava and Specksim in their shared SDK environment:

```sh
for TARGET in lava specksim; do
  docker run --rm --network none --cpus 2 --memory 3g \
    -v "$SDK_ENV:/env:ro" -v "$CODE:/code:ro" \
    -v "$API:/api:ro" -v "$RUN:/work" \
    atlas-neuro-runtime:20261013 /code/sdk_worker.py \
    --target "$TARGET" --runtime /api/neuromorphic_runtime.py --run /work
done
```

Run the additional count targets in their separate Python 3.12 environments:

```sh
PYTHONPATH='' PYTHONNOUSERSITE=1 /absolute/path/to/spinnaker2-env/bin/python \
  "$CODE/sdk_worker.py" --target spinnaker2 --runtime "$API/neuromorphic_runtime.py" --run "$RUN"
PYTHONPATH='' PYTHONNOUSERSITE=1 /absolute/path/to/xylo-env/bin/python \
  "$CODE/sdk_worker.py" --target xylosim --runtime "$API/neuromorphic_runtime.py" --run "$RUN"
python "$CODE/audit.py" --run "$RUN"
```

The audit fails on missing target results, changed hashes, any differing static
state, any differing Specksim remainder or a mismatch between Specksim batch
sizes 2 and 5. It retains per-case results, not only a combined pass flag.
The all-zero SpiNNaker2 count model exercises its analytical zero fast path;
this is explicitly marked as skipping SDK execution in the audit.

## Temporal coverage and compatibility check

```sh
python brian2-rust/examples/neuromorphic/spinnaker2/compare.py --coverage \
  --runner "$B2_RUNNER" --sdk-python /absolute/path/to/spinnaker2-env/bin/python \
  --output output/neuromorphic/spinnaker2-coverage
python brian2-rust/examples/neuromorphic/xylo_temporal/compare.py \
  --suite coverage --chunk-steps 7 --runner "$B2_RUNNER" \
  --sdk-python /absolute/path/to/xylo-env/bin/python \
  --output output/neuromorphic/xylo-coverage
python "$CODE/check_compatibility.py" --runner "$B2_RUNNER" \
  --static-run "$RUN" --spinnaker-run output/neuromorphic/spinnaker2-coverage \
  --xylo-run output/neuromorphic/xylo-coverage --output "$RUN/compatibility.json"
```

The routing check verifies all 20 public target/profile combinations. It is
separate from SDK execution and does not certify arbitrary graphs within a
profile. Temporal checks start from zero state and compare the full run.
Xylo's chunk check preserves one SDK model across calls; the static sample
reset tests are a different contract.

Run the shared admission and runtime regressions:

```sh
B2_RUNNER="$B2_RUNNER" python -m pytest -q \
  brian2-rust/tests/test_neuromorphic.py \
  brian2-rust/tests/test_akida_b2ir.py \
  brian2-rust/tests/test_lava_dense_b2ir.py \
  brian2-rust/tests/test_spinnaker2_b2ir.py \
  brian2-rust/tests/test_xylo_b2ir.py
```
