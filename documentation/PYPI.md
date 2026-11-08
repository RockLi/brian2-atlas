# 正式 PyPI 发布

正式发布使用 `.github/workflows/release-pypi.yml`。合并 PR 到 `dev` 只更新代码，
不会上传 PyPI；只有推送与包版本一致的稳定版 `v*` 标签才会发布。当前
`0.2.0.dev1` 是已完成 TestPyPI 演练的开发版本，正式版本号需要在发布提交中更新。

## 正式账号权限

登录 [PyPI Publishing](https://pypi.org/manage/account/publishing/)，添加 GitHub
pending publisher。正式 PyPI 与 TestPyPI 的账号及授权相互独立。

| 字段 | 正式配置 |
| --- | --- |
| Project name | `brian2-atlas` |
| Owner | `RockLi` |
| Repository name | `brian2-atlas` |
| Workflow name | `release-pypi.yml` |
| Environment name | `pypi` |

若项目已存在，需要项目 Owner 在项目 Publishing 页面添加相同配置；按 PyPI
页面提示完成账号验证与双因素认证。无需创建或分享 API token。Pending publisher
在首次上传时创建项目，本身不预留包名。

GitHub 发布 job 使用名为 `pypi` 的 environment，只有该 job 具有 `id-token: write`。
建议在仓库 Settings → Environments 预建 `pypi`，将部署来源限制为 `v*` 标签，
并按团队发布流程设置 required reviewer。如果启用 reviewer，上传前需要该 reviewer
在 GitHub 批准。这项审批是仓库设置，不等于 PyPI 的 Trusted Publisher 配置。

## 发布前准备

1. 确定稳定版本，例如 `0.2.0`；更新 `pyproject.toml` 并运行 `uv lock`。
2. 将 `project.urls.Documentation` 改为
   `https://github.com/RockLi/brian2-atlas/blob/v0.2.0/documentation/README.md`，
   把 `README.pypi.md` 中的候选标签链接和 TestPyPI 安装说明改为正式版本内容。
   更新发布说明和适用的引用元数据；保留历史 `v0.1.0` 研究结果的版本边界。
3. 将发布提交通过 PR 合并到 `dev`，检查 frontend、CPU、分发和发布构建结果。
4. 建议先以相同稳定版本运行一次 TestPyPI 演练。已通过的 `0.2.0.dev1` 不等于
   最终版本的产物验证；版本、源代码或依赖变化后需要对最终产物重新验证。

正式流水线沿用 Linux x86_64 / CPython 3.12、3.13、3.14 的 manylinux 2.28
构建范围，同时提供 sdist。其他平台可能需要源码安装，macOS/Windows 发布 wheel
尚未加入该矩阵。基础 CPU wheel 使用不需要本地重编译 Rust；AOT 和部分后端另有
工具链要求，见 [安装说明](INSTALLATION.md)。

## 执行与验证

在版本提交已经合并、测试完成且正式发布已获授权后执行：

```sh
git fetch origin dev
git tag v0.2.0 RELEASE_COMMIT
git push origin refs/tags/v0.2.0
```

用经过验证的提交替换 `RELEASE_COMMIT`。工作流先检查标签/版本一致、版本不是
dev/rc/local，并验证标签提交是 `origin/dev` 的祖先；随后从 sdist 构建三个 wheel，
执行 auditwheel 修复与安装运行测试。通过后，发布 job 仅上传同一次运行生成的
产物，并记录 SHA-256。手动运行 workflow 或 PR 触发只构建，不上传。

上传后，三个验证 job 从正式 PyPI 重新下载文件，核对文件集合、索引哈希和实际
下载字节，再分别安装对应 wheel，执行 CPU/AOT 仿真、训练和 checkpoint 检查。
通过后才将发布标记为验证完成。保留 `publication-checksums` 和
`pypi-validation-*` artifacts 及工作流链接。

如果上传部分成功或验证失败，不移动旧标签、不覆盖同名文件。核对原始产物后
处理失败原因；需要重新构建时使用新版本，必要时 yank 有问题的旧版本。

用户安装正式版本时应新建虚拟环境，再运行 `python -m pip install brian2-atlas`。
Atlas 自带 `brian2`，不能与上游 `Brian2` distribution 共用环境。

参考：[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)。
