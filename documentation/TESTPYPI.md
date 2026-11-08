# TestPyPI 发布演练

首个候选版本为 `0.2.0.dev1`。它来自当前开发代码，不代表冻结的 `v0.1.0`
研究版本，也不是正式 PyPI 发布。本次上传范围为源码包和 Linux x86_64 的
CPython 3.12、3.13、3.14 wheel，最低 glibc 版本为 2.28。

## 账号配置

TestPyPI 与正式 PyPI 的账号、项目权限和发布者配置相互独立。登录
[TestPyPI 的 Publishing 页面](https://test.pypi.org/manage/account/publishing/)，
添加 GitHub 的 pending publisher：

| 字段 | 值 |
| --- | --- |
| PyPI project name | `brian2-atlas` |
| Owner | `RockLi` |
| Repository name | `brian2-atlas` |
| Workflow name | `release-testpypi.yml` |
| Environment name | `testpypi` |

若项目已存在，应由项目 Owner 在该项目的 Publishing 页面添加同样的配置。
按 TestPyPI 的账号提示完成邮箱验证和双因素认证。无需生成或分享 API token。
Pending publisher 在首次成功上传时创建项目，本身不预留包名。

GitHub 工作流使用 `testpypi` environment，发布 job 单独具有 `id-token: write`。
可以在仓库 Settings → Environments 中预建此 environment，并限制允许部署的标签为
`testpypi-v*`。如果设置了 required reviewer，发布执行到该处后需要 reviewer 批准。
生产 PyPI 的权限本次不需要配置。

## 构建和发布触发条件

工作流为 [release-testpypi.yml](../.github/workflows/release-testpypi.yml)：

1. 推送 `codex/testpypi-release` 分支会构建源码包和三个 Linux wheel。
2. 每个 wheel 都在固定的 manylinux 2.28 容器中构建，经过 auditwheel 修复、
   包内执行器 SHA-256/RECORD 更新、实际安装与运行检查。
3. 仅 `testpypi-v版本号` 标签会触发上传；标签必须与 `pyproject.toml` 的版本完全一致。
4. 发布 job 只下载同一次运行中通过测试的构建产物，通过 OIDC 上传至 TestPyPI。
5. 上传后的三个验证 job 下载同一版本的全部文件并核对原始构建哈希，然后分别安装
   与当前 Python 匹配的 wheel。运行依赖来自正式 PyPI，Atlas wheel 来自 TestPyPI。

在本地提交、分支 CI 通过、账号配置完成后，发布者在选定提交上执行：

```sh
git tag testpypi-v0.2.0.dev1 RELEASE_COMMIT
git push origin refs/tags/testpypi-v0.2.0.dev1
```

`RELEASE_COMMIT` 必须替换为已经验证的提交。不要移动已发布标签。第一次上传若只
完成部分文件，先检查失败原因；重新构建应使用新的开发版本，不能覆盖原有文件。
仅重试尚未上传成功的发布操作时也要保留并核对原始产物。

工作流没有正式 PyPI 上传步骤。生产发布需要单独确定正式版本、覆盖的平台和权限。

## 本地重现 Linux 构建

安装 Docker、Python 构建工具以及 `cibuildwheel==4.3.5`，在仓库根目录执行：

```sh
python -m build --sdist --outdir build/testpypi/source-dist
python -m cibuildwheel --platform linux \
  --output-dir build/testpypi/wheelhouse build/testpypi/source-dist/*.tar.gz
python -m twine check --strict build/testpypi/wheelhouse/*.whl
```

输出目录应只包含当前候选版本。用 `CIBW_BUILD=cp312-manylinux_x86_64` 可仅构建
一个解释器版本。Apple Silicon 上 Docker 可通过仿真构建 x86_64；发布 CI 使用
原生 x86_64 runner。Rust 安装在容器内，不会依赖宿主机的 Rust 构建目录。

## 验证已上传版本

在 Linux x86_64、CPython 3.12–3.14 环境中，可以先从 TestPyPI 下载 Atlas wheel，
再让正式 PyPI 提供运行依赖：

```sh
python -m pip download --index-url https://test.pypi.org/simple/ \
  --no-deps --only-binary=:all: --dest downloaded brian2-atlas==0.2.0.dev1
python -m venv installed
installed/bin/python -m pip install --index-url https://pypi.org/simple/ downloaded/*.whl
installed/bin/python dev/atlas/test_wheel.py --report installed-testpypi.json
```

完整安装检查需要 Rust 1.98.1 和 C/C++ 编译器，以验证 AOT 和 C host kernel。
只运行基础 CPU 仿真和训练示例不需要本地重建 Rust 执行器。

发布日志保留上传文件的 SHA-256。`publication-checksums` artifact 保存校验值，
`testpypi-validation-*` artifacts 保存各 Python 版本的运行报告。全部验证完成后，
才把该次 TestPyPI 演练标记为通过。

参考：[PyPI Trusted Publishing 项目创建说明](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)。
