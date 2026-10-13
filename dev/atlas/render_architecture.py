#!/usr/bin/env python3
"""Render the README architecture figures. Requires matplotlib; no Atlas runtime.

Usage: MPLCONFIGDIR=/tmp/atlas-mpl python dev/atlas/render_architecture.py
English always renders; Chinese also renders when --cjk-font or the macOS
Arial Unicode font is available. Generated files live in brian2-rust/architecture/.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

W, H = 1800, 1370
INK, MUTED, LINE = "#172B42", "#56687B", "#B9C8D6"
BLUE, PURPLE, TEAL = "#2265B5", "#7554AF", "#087F78"


def render(out, lang, font_path=None):
    zh = lang == "zh"
    font = FontProperties(fname=font_path) if font_path else FontProperties(family="DejaVu Sans")
    fig = plt.figure(figsize=(18, 13.7), dpi=120, facecolor="#F7F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0)); ax.axis("off")
    def t(x, y, en, cn=None, size=15, color=INK, weight="normal", ha="left"):
        return ax.text(x, y, cn if zh and cn is not None else en,
                       fontsize=size, color=color, fontproperties=font,
                       weight=weight, ha=ha, va="top", linespacing=1.5)
    def box(x, y, w, h, fill="white", edge=LINE, dashed=False):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0,rounding_size=16",
                                  linewidth=1.15, edgecolor=edge, facecolor=fill,
                                  linestyle=(0,(5,4)) if dashed else "solid"))
    def arrow(points, color=LINE):
        for a,b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]], color=color, lw=1.6)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1], arrowstyle="-|>",
                                    mutation_scale=15, lw=1.6, color=color))
    def rule(x,y,w): ax.plot([x,x+w],[y,y], color="#DBE3EC", lw=1)

    t(70, 36, "ATLAS", size=31, weight="bold")
    t(70, 90, "Simulation · Training · Conversion", "仿真 · 训练 · 转换", size=24)
    t(70, 137, "One modelling frontend, three explicit execution paths",
      "统一的建模入口，三条职责清晰的执行路径", color=MUTED, size=14)

    box(70,195,1110,112)
    t(95,212,"Brian2 modelling frontend", "Brian2 建模前端",size=19,weight="bold")
    t(95,251,"Equations · state · topology · clocks · events · parameters",
      "方程 · 状态 · 网络连接 · 时钟 · 事件 · 参数", size=15,color=MUTED)
    box(1250,195,480,112,fill="#EEF3F8")
    t(1273,212,"External TensorFlow training", "外部 TensorFlow 训练",size=18,weight="bold")
    t(1273,251,"Frozen weights → explicit Brian2 model",
      "冻结权重 → 显式构建 Brian2 模型",size=14,color=MUTED)
    arrow([(1250,251),(1182,251)])

    box(70,357,500,112,fill="#F0EAF9",edge="#C5B5DC")
    t(95,375,"Training lowering", "训练语义编译",size=19,color=PURPLE,weight="bold")
    t(95,414,"Dedicated, versioned training plans",
      "独立、带版本的训练计划",size=15)
    box(625,357,1105,112,fill="#EAF2FC",edge="#AFC8E8")
    t(650,373,"AtlasIR  ·  b2ir-v1",size=20,color=BLUE,weight="bold")
    t(650,411,"Typed state + schedule / event semantics · canonical hashes · native validation",
      "带类型的状态 + 调度 / 事件语义 · 规范化哈希 · 原生校验",size=15)
    arrow([(320,307),(320,355)],PURPLE)
    arrow([(905,307),(905,355)],BLUE)
    arrow([(320,469),(320,517)],PURPLE)
    arrow([(875,469),(875,517)],BLUE)
    arrow([(1457,469),(1457,517)],TEAL)

    for x,w,c,bg in [(70,500,PURPLE,"#FCFAFF"),(625,500,BLUE,"#F8FBFF"),(1180,550,TEAL,"#F6FCFB")]:
        box(x,520,w,595,fill=bg,edge=c)
    t(95,544,"TRAINING", "训练",size=22,color=PURPLE,weight="bold")
    t(650,544,"SIMULATION", "仿真",size=22,color=BLUE,weight="bold")
    t(1205,544,"CONVERSION", "转换",size=22,color=TEAL,weight="bold")

    t(95,597,"NativeLIFTrainer / supported Brian lowering",
      "NativeLIFTrainer / 支持子集的 Brian 编译",size=13,color=MUTED)
    box(95,638,450,94,fill="#F0EAF9",edge="#D5C7E7")
    t(115,651,"b2-train",size=19,weight="bold")
    t(115,689,"Forward → surrogate gradients → optimizer",
      "前向 → 替代梯度反传 → 优化器",size=13)
    t(95,766,"CPU / CUDA / Apple Metal",size=17,weight="bold")
    t(95,811,"MPI training · same-host validation",
      "MPI 训练 · 已验证同机执行",size=15)
    t(95,856,"BPTT / TBPTT · bounded semantics",
      "BPTT / TBPTT · 支持限定语义",size=15)
    rule(95,914,450)
    t(95,939,"Result", "输出",size=14,color=PURPLE,weight="bold")
    t(95,974,"Weights + optimizer / state checkpoints",
      "权重 + 优化器 / 状态检查点",size=15)
    t(95,1035,"Training plans have a separate contract\nfrom the frozen AtlasIR simulation ABI.",
      "训练计划具有独立协议，\n与冻结的 AtlasIR 仿真 ABI 分开版本化。",size=13,color=MUTED)

    t(650,597,"AtlasDevice / b2-runner + browser exporter",
      "AtlasDevice / b2-runner + 浏览器导出",size=13,color=MUTED)
    for y,label,detail,cn_label,cn_detail in [
        (649,"CPU","Reference + specialized AOT","CPU","参考执行器 + 模型特化 AOT"),
        (720,"GPU","CUDA / Apple Metal · float32","GPU","CUDA / Apple Metal · float32"),
        (791,"Browser","WASM Workers / WebGPU","浏览器","WASM Workers / WebGPU"),
        (862,"MPI + heterogeneous execution","CPU ranks + GPU state-update offload","MPI + 异构执行","CPU 进程 + GPU 状态更新卸载"),
    ]:
        t(650,y,label,cn_label,size=16,weight="bold")
        t(650,y+30,detail,cn_detail,size=14,color=MUTED)
    rule(650,932,450)
    t(650,951,"Result", "输出",size=14,color=BLUE,weight="bold")
    t(650,982,"State trajectories · spikes · monitors",
      "状态轨迹 · 脉冲 · 监视器记录",size=15)
    t(650,1035,"Heterogeneous path: host event ordering,\nf64 storage + f32 GPU state updates.",
      "异构路径：主机保持事件顺序与 f64 存储，\nGPU 以 f32 执行状态更新。",size=13,color=MUTED)

    t(1205,597,"convert_ir / export_ir / run_simulator",size=14,color=MUTED)
    box(1205,638,500,114,fill="#E6F4F1",edge="#A9D3CA")
    t(1225,653,"Semantic profile + resource checks",
      "语义子集 + 数值 / 资源约束检查",size=16,weight="bold")
    t(1225,697,"Target lowering → private plan → SDK",
      "目标编译 → 后端专用计划 → SDK",size=15)
    t(1205,782,"Six software targets", "六个软件目标",size=16,weight="bold")
    for x,y,en,cn in [(1205,824,"Akida","Akida"),(1465,824,"Lava / Loihi","Lava / Loihi"),
                     (1205,875,"Specksim","Specksim"),(1465,875,"py-spinnaker2","py-spinnaker2"),
                     (1205,926,"XyloSim","XyloSim"),(1465,926,"DynapSim","DynapSim")]:
        box(x,y,240,39,fill="white",edge="#C7DDD8")
        t(x+14,y+9,en,cn,size=14)
    t(1205,989,"SDK states / outputs / spike times",
      "SDK 状态 / 输出 / 脉冲时刻",size=15)
    t(1205,1035,"Target-specific subsets; unsupported models\nare rejected. Current execution is on CPU.",
      "各目标支持不同子集；不支持的模型拒绝转换。\n当前验证均使用 CPU 上的软件 SDK。",size=13,color=MUTED)

    box(625,1152,1105,76,fill="#EAF1F8",edge="#BECEDC")
    arrow([(875,1115),(875,1150)],BLUE)
    arrow([(1457,1115),(1457,1150)],TEAL)
    t(650,1165,"Conversion validation  ·  Atlas reference ↔ target SDK",
      "转换验证  ·  Atlas 参考执行 ↔ 目标 SDK",size=17,weight="bold")
    t(650,1198,"Compare states, outputs and spike ticks; exact equality or an explicit profile tolerance.",
      "对比状态、输出与脉冲时刻；按语义子集要求逐位一致或满足明确误差界限。",size=13,color=MUTED)
    box(70,1152,500,76,fill="#FFFFFF",edge=LINE,dashed=True)
    t(91,1166,"No general training-checkpoint → chip bridge",
      "尚无通用的训练检查点 → 芯片转换通路",size=13,weight="bold")
    t(91,1198,"Conversion needs an admitted source model.",
      "转换必须满足目标支持的源模型语义。",size=13,color=MUTED)

    t(70,1260,"Verification boundary", "验证边界",size=14,weight="bold")
    t(70,1292,"Native CPU/GPU and same-host MPI: bounded validation. Cross-host / multi-GPU: not qualified.",
      "原生 CPU/GPU 与同机 MPI：已完成限定范围验证。跨主机 / 多物理 GPU：尚未完成验收。",size=13,color=MUTED)
    t(70,1321,"Neuromorphic: software conversion and inference. Physical-chip placement, execution, power and latency: not verified.",
      "神经形态：当前验证软件转换与推理。实体芯片映射、执行、功耗与延迟：尚未验证。",size=13,color=MUTED)
    out.mkdir(parents=True,exist_ok=True)
    stem = "atlas-architecture" + ("-zh" if zh else "")
    for ext in ("svg","png"):
        fig.savefig(out/f"{stem}.{ext}",dpi=140,facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[2]/"brian2-rust"/"architecture")
    parser.add_argument("--cjk-font",type=Path,default=Path("/Library/Fonts/Arial Unicode.ttf"))
    args = parser.parse_args()
    render(args.output_dir,"en")
    if args.cjk_font.is_file(): render(args.output_dir,"zh",str(args.cjk_font))
    else: print("Chinese rendering skipped: pass --cjk-font with a CJK font file.")

if __name__ == "__main__": main()
