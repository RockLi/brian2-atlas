#!/usr/bin/env python3
"""Render the README architecture figures. Requires matplotlib; no Atlas runtime.

Usage: MPLCONFIGDIR=/tmp/atlas-mpl python dev/atlas/render_architecture.py
Generated English figures live in brian2-rust/architecture/.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "atlas-architecture-v1"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

W, H = 1800, 1370
INK, MUTED, LINE = "#172B42", "#56687B", "#B9C8D6"
BLUE, PURPLE, TEAL = "#2265B5", "#7554AF", "#087F78"


def render(out):
    font = FontProperties(family="DejaVu Sans")
    fig = plt.figure(figsize=(18, 13.7), dpi=120, facecolor="#F7F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0)); ax.axis("off")
    def t(x, y, text, size=15, color=INK, weight="normal", ha="left"):
        return ax.text(x, y, text,
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
    t(70, 90, "Simulation · Training · Conversion", size=24)
    t(70, 137, "One modelling frontend, three explicit execution paths", color=MUTED, size=14)

    box(70,195,1110,112)
    t(95,212,"Brian2 modelling frontend",size=19,weight="bold")
    t(95,251,"Equations · state · topology · clocks · events · parameters", size=15,color=MUTED)
    box(1250,195,480,112,fill="#EEF3F8")
    t(1273,212,"External TensorFlow training",size=18,weight="bold")
    t(1273,251,"Frozen weights → explicit Brian2 model",size=14,color=MUTED)
    arrow([(1250,251),(1182,251)])

    box(70,357,500,112,fill="#F0EAF9",edge="#C5B5DC")
    t(95,375,"Training lowering",size=19,color=PURPLE,weight="bold")
    t(95,414,"Dedicated, versioned training plans",size=15)
    box(625,357,1105,112,fill="#EAF2FC",edge="#AFC8E8")
    t(650,373,"AtlasIR v1",size=20,color=BLUE,weight="bold")
    t(650,411,"Typed state + schedule / event semantics · canonical hashes · native validation",size=15)
    arrow([(320,307),(320,355)],PURPLE)
    arrow([(905,307),(905,355)],BLUE)
    arrow([(320,469),(320,517)],PURPLE)
    arrow([(875,469),(875,517)],BLUE)
    arrow([(1457,469),(1457,517)],TEAL)

    for x,w,c,bg in [(70,500,PURPLE,"#FCFAFF"),(625,500,BLUE,"#F8FBFF"),(1180,550,TEAL,"#F6FCFB")]:
        box(x,520,w,595,fill=bg,edge=c)
    t(95,544,"TRAINING",size=22,color=PURPLE,weight="bold")
    t(650,544,"SIMULATION",size=22,color=BLUE,weight="bold")
    t(1205,544,"CONVERSION",size=22,color=TEAL,weight="bold")

    t(95,597,"NativeLIFTrainer / NativeXyloTrainer",size=13,color=MUTED)
    box(95,638,450,94,fill="#F0EAF9",edge="#D5C7E7")
    t(115,651,"b2-train",size=19,weight="bold")
    t(115,689,"Forward → surrogate gradients → optimizer",size=13)
    t(95,766,"CPU / CUDA / Apple Metal",size=17,weight="bold")
    t(95,811,"MPI training · same-host validation",size=15)
    t(95,856,"BPTT / TBPTT · bounded semantics",size=15)
    t(95,889,"Xylo integer plan: CPU only",size=13,color=MUTED)
    rule(95,914,450)
    t(95,939,"Result",size=14,color=PURPLE,weight="bold")
    t(95,974,"Weights + optimizer / state checkpoints",size=15)
    t(95,1035,"Training plans have a separate contract\nfrom the frozen AtlasIR simulation ABI.",size=13,color=MUTED)

    t(650,597,"AtlasDevice / b2-runner + browser exporter",size=13,color=MUTED)
    for y,label,detail in [
        (649,"CPU","Reference + specialized AOT"),
        (720,"GPU","CUDA / Apple Metal · float32"),
        (791,"Browser","WASM Workers / WebGPU"),
        (862,"MPI + heterogeneous execution","CPU ranks + GPU state-update offload"),
    ]:
        t(650,y,label,size=16,weight="bold")
        t(650,y+30,detail,size=14,color=MUTED)
    rule(650,932,450)
    t(650,951,"Result",size=14,color=BLUE,weight="bold")
    t(650,982,"State trajectories · spikes · monitors",size=15)
    t(650,1035,"Heterogeneous path: host event ordering,\nf64 storage + f32 GPU state updates.",size=13,color=MUTED)

    t(1205,597,"convert_ir / export_ir / run_simulator",size=14,color=MUTED)
    box(1205,638,500,114,fill="#E6F4F1",edge="#A9D3CA")
    t(1225,653,"Semantic profile + resource checks",size=16,weight="bold")
    t(1225,697,"Target lowering → private plan → SDK",size=15)
    t(1205,782,"Six software targets",size=16,weight="bold")
    for x,y,label in [(1205,824,"Akida"),(1465,824,"Lava / Loihi"),
                      (1205,875,"Specksim"),(1465,875,"py-spinnaker2"),
                      (1205,926,"XyloSim"),(1465,926,"DynapSim")]:
        box(x,y,240,39,fill="white",edge="#C7DDD8")
        t(x+14,y+9,label,size=14)
    t(1205,989,"SDK states / outputs / spike times",size=15)
    t(1205,1035,"Target-specific subsets; unsupported models\nare rejected. Current execution is on CPU.",size=13,color=MUTED)

    box(625,1152,1105,76,fill="#EAF1F8",edge="#BECEDC")
    arrow([(875,1115),(875,1150)],BLUE)
    arrow([(1457,1115),(1457,1150)],TEAL)
    t(650,1165,"Conversion validation  ·  Atlas reference ↔ target SDK",size=17,weight="bold")
    t(650,1198,"Compare states, outputs and spike ticks; exact equality or an explicit profile tolerance.",size=13,color=MUTED)
    arrow([(320,1115),(320,1150)],PURPLE)
    box(70,1152,500,76,fill="#FFFFFF",edge=LINE,dashed=True)
    t(91,1166,"Native weights → AtlasIR → target",size=13,weight="bold")
    t(91,1198,"Exact LIF→SpiNNaker2 / integer→Xylo subsets.",size=13,color=MUTED)

    t(70,1260,"Verification boundary",size=14,weight="bold")
    t(70,1292,"Native CPU/GPU and same-host MPI: bounded validation. Cross-host / multi-GPU: not qualified.",size=13,color=MUTED)
    t(70,1321,"Neuromorphic: software conversion and inference. Physical-chip placement, execution, power and latency: not verified.",size=13,color=MUTED)
    out.mkdir(parents=True,exist_ok=True)
    stem = "atlas-architecture"
    for ext in ("svg","png"):
        path = out/f"{stem}.{ext}"
        fig.savefig(path,dpi=140,facecolor=fig.get_facecolor(),
                    metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[2]/"brian2-rust"/"architecture")
    args = parser.parse_args()
    render(args.output_dir)

if __name__ == "__main__": main()
