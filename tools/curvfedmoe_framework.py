#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CurvFedMoE framework figure reproduction.

Run from:
    ~/Project/clean

Outputs will be written to:
    ./curvfedmoe/

Generated files:
    curvfedmoe_reproduced.png
    curvfedmoe_reproduced.pdf
    curvfedmoe_reproduced.svg
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe for remote/headless Linux servers

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import (
    FancyBboxPatch,
    Rectangle,
    Polygon,
    FancyArrowPatch,
    Circle,
)
from matplotlib.colors import to_rgb


# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = Path("curvfedmoe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Global style
# ============================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stixsans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# Palette
# ============================================================
NAVY = "#1F2F5B"
BLUE = "#4B8FC5"
BLUE_BORDER = "#6CA6D3"
BLUE_BG = "#EEF7FD"

PURPLE = "#7551B5"
PURPLE_BORDER = "#9F83D3"
PURPLE_BG = "#F6F1FC"

GREEN_BORDER = "#8CCB9A"
GREEN_BG = "#F5FBF5"

TEXT = "#1E2B45"
MUTED = "#707A8B"
LIGHT_GREY = "#D6DBE2"
ROUTER_BG = "#D9DDE5"
ROUTER_BORDER = "#959DA9"

EXPERT_COLORS = ["#4D9BD3", "#E99A52", "#A67BDE", "#5EBE78"]

RED = "#D94740"
DIRECTION_BLUE = "#2F75B5"


# ============================================================
# Helpers
# ============================================================
def lighten(color, amount=0.15):
    r, g, b = to_rgb(color)
    return (
        1 - (1 - r) * (1 - amount),
        1 - (1 - g) * (1 - amount),
        1 - (1 - b) * (1 - amount),
    )


def darken(color, amount=0.15):
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def rounded_box(
    ax, x, y, w, h,
    facecolor="white",
    edgecolor="black",
    linewidth=1.4,
    radius=0.55,
    zorder=1,
    alpha=1.0,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(
    ax, x, y, s,
    fontsize=11,
    weight="normal",
    color=TEXT,
    ha="left",
    va="center",
    zorder=20,
):
    ax.text(
        x, y, s,
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def arrow(
    ax,
    start,
    end,
    color=NAVY,
    linewidth=1.6,
    mutation_scale=13,
    style="-|>",
    connectionstyle="arc3",
    zorder=12,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_3d_expert(ax, x, y, w=2.35, h=1.75, depth=0.55, color="#4D9BD3", zorder=12):
    """Small pseudo-3D expert parameter block."""
    edge = darken(color, 0.28)

    # front
    ax.add_patch(Rectangle(
        (x, y), w, h,
        facecolor=color,
        edgecolor=edge,
        linewidth=0.9,
        zorder=zorder + 2,
    ))

    # right
    ax.add_patch(Polygon(
        [
            (x + w, y),
            (x + w + depth, y + depth * 0.5),
            (x + w + depth, y + h + depth * 0.5),
            (x + w, y + h),
        ],
        closed=True,
        facecolor=darken(color, 0.08),
        edgecolor=edge,
        linewidth=0.9,
        zorder=zorder + 1,
    ))

    # top
    ax.add_patch(Polygon(
        [
            (x, y + h),
            (x + depth, y + h + depth * 0.5),
            (x + w + depth, y + h + depth * 0.5),
            (x + w, y + h),
        ],
        closed=True,
        facecolor=lighten(color, 0.18),
        edgecolor=edge,
        linewidth=0.9,
        zorder=zorder + 3,
    ))

    # central split, matching reference style
    ax.plot(
        [x + w * 0.50, x + w * 0.50],
        [y, y + h],
        color=edge,
        linewidth=0.55,
        alpha=0.65,
        zorder=zorder + 4,
    )


def draw_matrix_icon(
    ax, x, y, size=2.55, n=3,
    base_color="#A9D2EF",
    edgecolor="#6B9CB8",
):
    cell = size / n
    for r in range(n):
        for c in range(n):
            shade = 0.03 + 0.09 * ((r + 2 * c) % 3)
            ax.add_patch(Rectangle(
                (x + c * cell, y + r * cell),
                cell, cell,
                facecolor=lighten(base_color, shade),
                edgecolor=edgecolor,
                linewidth=0.65,
                zorder=14,
            ))


def bracket_right(ax, x, y0, y1, width=0.9, color=TEXT, linewidth=1.25):
    ax.plot([x, x + width], [y1, y1], color=color, linewidth=linewidth, zorder=12)
    ax.plot([x + width, x + width], [y0, y1], color=color, linewidth=linewidth, zorder=12)
    ax.plot([x, x + width], [y0, y0], color=color, linewidth=linewidth, zorder=12)


# ============================================================
# Major regions
# ============================================================
def draw_outer_panels(ax):
    # Client-side area
    rounded_box(
        ax, 0.8, 2.9, 36.2, 43.5,
        facecolor=BLUE_BG,
        edgecolor=BLUE_BORDER,
        linewidth=1.55,
        radius=0.75,
        zorder=1,
    )

    # Server-side area
    rounded_box(
        ax, 48.7, 3.6, 50.1, 42.3,
        facecolor=PURPLE_BG,
        edgecolor=PURPLE,
        linewidth=1.55,
        radius=0.75,
        zorder=1,
    )

    text(ax, 1.9, 44.5, "Client-side", fontsize=21, weight="bold", color=NAVY)
    text(ax, 1.9, 42.5, "(multiple clients)", fontsize=15, weight="bold", color=NAVY)

    text(ax, 49.8, 44.5, "Server-side", fontsize=21, weight="bold", color=NAVY)


def draw_client_tabs(ax):
    rounded_box(
        ax, 19.1, 40.2, 10.7, 4.15,
        facecolor="#F4F8FC",
        edgecolor=BLUE_BORDER,
        linewidth=1.25,
        radius=0.55,
        zorder=4,
    )
    rounded_box(
        ax, 25.5, 41.2, 10.7, 4.15,
        facecolor="#F4F8FC",
        edgecolor=BLUE_BORDER,
        linewidth=1.25,
        radius=0.55,
        zorder=5,
    )
    rounded_box(
        ax, 31.9, 42.2, 10.7, 4.15,
        facecolor="#F4F8FC",
        edgecolor=BLUE_BORDER,
        linewidth=1.25,
        radius=0.55,
        zorder=6,
    )

    text(ax, 24.4, 42.3, "Client 1", fontsize=13, weight="bold", ha="center")
    text(ax, 30.8, 43.3, "Client 2", fontsize=12, ha="center")
    text(ax, 37.2, 44.3, "Client N", fontsize=12, ha="center")
    text(ax, 40.8, 41.7, "⋯", fontsize=17, ha="center")


# ============================================================
# Client-side: Local MoE
# ============================================================
def draw_local_moe(ax):
    x0, y0, w, h = 2.0, 25.6, 33.3, 14.8

    rounded_box(
        ax, x0, y0, w, h,
        facecolor="#F7FBFE",
        edgecolor="#80BFE4",
        linewidth=1.25,
        radius=0.55,
        zorder=3,
    )

    text(ax, x0 + 1.0, y0 + h - 1.7, "Local MoE training", fontsize=15, weight="bold")
    text(ax, x0 + 13.7, y0 + h - 1.7, "(top-2 routing)", fontsize=14)

    # Input x
    text(ax, x0 + 2.0, y0 + 7.5, r"$x$", fontsize=20, weight="bold")
    arrow(ax, (x0 + 3.2, y0 + 7.5), (x0 + 6.4, y0 + 7.5), color=NAVY, linewidth=1.5)

    # Router
    rounded_box(
        ax, x0 + 6.5, y0 + 5.7, 5.3, 3.7,
        facecolor=ROUTER_BG,
        edgecolor=ROUTER_BORDER,
        linewidth=1.1,
        radius=0.35,
        zorder=8,
    )
    text(ax, x0 + 9.15, y0 + 7.55, "Router", fontsize=12, weight="bold", ha="center")

    # Bus line
    ax.plot(
        [x0 + 12.0, x0 + 13.3],
        [y0 + 7.5, y0 + 7.5],
        color=NAVY,
        linewidth=1.5,
        zorder=9,
    )
    ax.plot(
        [x0 + 13.3, x0 + 13.3],
        [y0 + 3.2, y0 + 11.0],
        color=NAVY,
        linewidth=1.35,
        zorder=9,
    )

    expert_ys = [y0 + 9.8, y0 + 7.2, y0 + 4.6, y0 + 2.0]

    # first two active, lower two inactive/dashed
    for idx, yy in enumerate(expert_ys):
        active = idx < 2
        ax.plot(
            [x0 + 13.3, x0 + 14.6],
            [yy + 0.8, yy + 0.8],
            color=EXPERT_COLORS[idx] if active else "#ADB4BF",
            linewidth=1.35 if active else 1.1,
            linestyle="-" if active else "--",
            zorder=9,
        )
        arrow(
            ax,
            (x0 + 14.6, yy + 0.8),
            (x0 + 17.7, yy + 0.8),
            color=EXPERT_COLORS[idx] if active else "#ADB4BF",
            linewidth=1.35 if active else 1.05,
            mutation_scale=11,
        )
        draw_3d_expert(
            ax,
            x0 + 18.2,
            yy,
            w=2.7,
            h=1.6,
            depth=0.55,
            color=EXPERT_COLORS[idx],
        )

    # Right bracket for expert stack
    bracket_right(
        ax,
        x0 + 24.2,
        expert_ys[-1] + 0.2,
        expert_ys[0] + 2.0,
        width=1.0,
        color="#353D4A",
        linewidth=1.15,
    )

    text(ax, x0 + 20.0, y0 + 1.0, "Top-2 active", fontsize=12, ha="center")


# ============================================================
# Client-side: TKFAC
# ============================================================
def draw_tkfac(ax):
    x0, y0, w, h = 2.0, 6.4, 33.3, 15.4

    rounded_box(
        ax, x0, y0, w, h,
        facecolor=GREEN_BG,
        edgecolor=GREEN_BORDER,
        linewidth=1.25,
        radius=0.55,
        zorder=3,
    )

    text(ax, x0 + 1.0, y0 + h - 1.6, "TKFAC curvature estimation", fontsize=14.5, weight="bold")
    text(ax, x0 + 16.0, y0 + h - 1.6, "(per expert layer)", fontsize=13.5)

    # Activations A
    rounded_box(
        ax, x0 + 1.1, y0 + 7.7, 8.1, 4.0,
        facecolor="#DCEFFC",
        edgecolor="#7CB6DE",
        linewidth=1.0,
        radius=0.35,
        zorder=8,
    )
    text(ax, x0 + 5.15, y0 + 10.3, "Activations", fontsize=11.5, weight="bold", ha="center")
    text(ax, x0 + 5.15, y0 + 8.6, r"$A$", fontsize=21, ha="center")

    # Output gradients G
    rounded_box(
        ax, x0 + 1.1, y0 + 2.9, 8.1, 4.0,
        facecolor="#FCE4E4",
        edgecolor="#E58F8F",
        linewidth=1.0,
        radius=0.35,
        zorder=8,
    )
    text(ax, x0 + 5.15, y0 + 5.75, "Output", fontsize=11.1, weight="bold", ha="center")
    text(ax, x0 + 5.15, y0 + 4.7, "gradients", fontsize=11.1, weight="bold", ha="center")
    text(ax, x0 + 5.15, y0 + 3.4, r"$G$", fontsize=19, ha="center")

    # TKFAC central box
    rounded_box(
        ax, x0 + 12.8, y0 + 6.15, 5.4, 3.6,
        facecolor="#F8E9A9",
        edgecolor="#D6BA62",
        linewidth=1.0,
        radius=0.32,
        zorder=8,
    )
    text(ax, x0 + 15.5, y0 + 7.95, "TKFAC", fontsize=12.5, weight="bold", ha="center")

    arrow(
        ax,
        (x0 + 9.3, y0 + 9.7),
        (x0 + 12.8, y0 + 8.3),
        color="#3F536A",
        linewidth=1.25,
        mutation_scale=10,
    )
    arrow(
        ax,
        (x0 + 9.3, y0 + 4.9),
        (x0 + 12.8, y0 + 7.35),
        color="#3F536A",
        linewidth=1.25,
        mutation_scale=10,
    )

    # Outputs rho / Phi / Psi
    arrow(
        ax,
        (x0 + 18.2, y0 + 8.0),
        (x0 + 21.4, y0 + 10.6),
        color="#3F536A",
        linewidth=1.2,
        mutation_scale=10,
    )
    arrow(
        ax,
        (x0 + 18.2, y0 + 8.0),
        (x0 + 21.4, y0 + 7.8),
        color="#3F536A",
        linewidth=1.2,
        mutation_scale=10,
    )
    arrow(
        ax,
        (x0 + 18.2, y0 + 8.0),
        (x0 + 21.4, y0 + 4.5),
        color="#3F536A",
        linewidth=1.2,
        mutation_scale=10,
    )

    # rho
    rho = Circle(
        (x0 + 23.2, y0 + 10.75),
        1.25,
        facecolor="#EDE4FB",
        edgecolor="#B296DB",
        linewidth=1.0,
        zorder=12,
    )
    ax.add_patch(rho)
    text(ax, x0 + 23.2, y0 + 10.75, r"$\rho$", fontsize=22, ha="center")
    text(ax, x0 + 26.0, y0 + 10.75, "Scale", fontsize=12.5)

    # Phi
    draw_matrix_icon(
        ax, x0 + 21.2, y0 + 6.4,
        size=2.45,
        base_color="#A7D2EE",
        edgecolor="#6999B6",
    )
    text(ax, x0 + 24.35, y0 + 7.6, r"$\mathbf{\Phi}$", fontsize=19, weight="bold")

    # Psi
    draw_matrix_icon(
        ax, x0 + 21.2, y0 + 2.85,
        size=2.45,
        base_color="#F2BC82",
        edgecolor="#C78B50",
    )
    text(ax, x0 + 24.35, y0 + 4.05, r"$\mathbf{\Psi}$", fontsize=19, weight="bold")

    text(ax, x0 + 27.0, y0 + 6.0, "Directional\nstructure", fontsize=12.2)

    # Bottom equation
    text(
        ax,
        x0 + 11.2,
        y0 + 1.25,
        r"$\widehat{\mathbf{F}}=\rho(\mathbf{\Phi}\otimes\mathbf{\Psi})$",
        fontsize=18,
    )


# ============================================================
# Client -> Server upload
# ============================================================
def draw_upload(ax):
    # large light-blue arrow
    patch = FancyArrowPatch(
        (37.9, 28.0),
        (47.4, 28.0),
        arrowstyle="simple",
        mutation_scale=34,
        linewidth=0,
        facecolor="#4E8FBE",
        edgecolor="#4E8FBE",
        alpha=0.95,
        zorder=8,
    )
    ax.add_patch(patch)

    text(ax, 42.6, 31.5, "Parameters\n+ curvature", fontsize=12.5, weight="bold", ha="center")
    text(ax, 42.6, 24.6, r"$(n_i)$", fontsize=16, ha="center")

    # expert parameter mini-stack
    xs = [38.8, 40.2, 41.6, 43.0]
    for idx, xx in enumerate(xs):
        draw_3d_expert(
            ax,
            xx,
            18.7,
            w=0.88,
            h=1.95,
            depth=0.25,
            color=EXPERT_COLORS[idx],
            zorder=12,
        )

    text(
        ax,
        42.0,
        15.8,
        r"$\theta_i,\ \rho_i,\ \Phi_i,\ \Psi_i,\ n_i$",
        fontsize=15.3,
        ha="center",
    )


# ============================================================
# Server-side
# ============================================================
def draw_server_header(ax):
    rounded_box(
        ax, 49.5, 7.0, 48.1, 33.6,
        facecolor="#FBF8FE",
        edgecolor="#B399D6",
        linewidth=1.15,
        radius=0.55,
        zorder=3,
    )

    text(
        ax,
        73.55,
        39.0,
        "Curvature-guided aggregation",
        fontsize=18,
        weight="bold",
        ha="center",
    )


def draw_weighted_initialization(ax):
    x0, y0, w, h = 50.5, 27.0, 15.2, 9.3
    rounded_box(
        ax, x0, y0, w, h,
        facecolor="#FCFBFE",
        edgecolor="#B399D6",
        linewidth=1.0,
        radius=0.38,
        zorder=5,
    )

    text(ax, x0 + w/2, y0 + h - 1.25, "Weighted initialization", fontsize=11.8, weight="bold", ha="center")
    text(ax, x0 + w/2, y0 + h - 2.45, "(sample-size weighted)", fontsize=10.4, ha="center")

    block_xs = [x0 + 1.2, x0 + 4.0, x0 + 6.8, x0 + 11.7]
    for idx, xx in enumerate(block_xs):
        draw_3d_expert(
            ax, xx, y0 + 3.1,
            w=1.75, h=1.55, depth=0.38,
            color=EXPERT_COLORS[idx],
        )

    text(ax, x0 + 10.0, y0 + 4.0, "⋯", fontsize=18, ha="center")
    text(
        ax,
        x0 + w/2,
        y0 + 1.1,
        r"$\theta^{(0)}=\sum_i p_i\theta_i$",
        fontsize=15.3,
        ha="center",
    )


def draw_fisher_optimization(ax):
    x0, y0, w, h = 66.7, 27.0, 19.6, 9.3
    rounded_box(
        ax, x0, y0, w, h,
        facecolor="#FCFBFE",
        edgecolor="#B399D6",
        linewidth=1.0,
        radius=0.38,
        zorder=5,
    )

    text(ax, x0 + w/2, y0 + h - 1.25, "Fisher optimization", fontsize=11.8, weight="bold", ha="center")
    text(ax, x0 + w/2, y0 + h - 2.45, "(K steps)", fontsize=10.4, ha="center")

    rounded_box(
        ax, x0 + 1.15, y0 + 2.25, w - 2.3, 3.6,
        facecolor="#FAECEE",
        edgecolor="#E9C9D0",
        linewidth=0.9,
        radius=0.30,
        zorder=7,
    )

    text(
        ax,
        x0 + w/2,
        y0 + 4.05,
        r"$\min_{\theta}\ \frac{1}{2}\sum_i p_i"
        r"(\theta-\theta_i)^{\mathsf{T}}\widehat{\mathbf{F}}_i"
        r"(\theta-\theta_i)$",
        fontsize=13.6,
        ha="center",
    )

    text(ax, x0 + w/2, y0 + 1.05, r"$p_i$: sample-size weight", fontsize=10.6, ha="center")


def draw_global_experts(ax):
    x0, y0, w, h = 87.0, 27.0, 9.4, 9.3
    rounded_box(
        ax, x0, y0, w, h,
        facecolor="#FCFBFE",
        edgecolor="#B399D6",
        linewidth=1.0,
        radius=0.38,
        zorder=5,
    )

    text(ax, x0 + w/2, y0 + h - 1.3, "Global experts", fontsize=11.8, weight="bold", ha="center")

    ys = [y0 + 5.6, y0 + 3.7, y0 + 1.8, y0 - 0.1]
    for idx, yy in enumerate(ys):
        if idx < 3:
            draw_3d_expert(
                ax,
                x0 + 2.45,
                yy,
                w=2.35,
                h=1.35,
                depth=0.45,
                color=EXPERT_COLORS[idx],
            )
    text(ax, x0 + 6.8, y0 + 3.6, "⋯", fontsize=17, ha="center")
    draw_3d_expert(
        ax,
        x0 + 2.45,
        y0 + 0.05,
        w=2.35,
        h=1.35,
        depth=0.45,
        color=EXPERT_COLORS[3],
    )


def draw_direction_aware(ax):
    # divider
    ax.plot(
        [50.4, 96.6],
        [23.0, 23.0],
        color="#BFB4D6",
        linewidth=0.85,
        linestyle=(0, (4, 4)),
        zorder=5,
    )

    text(ax, 50.5, 21.1, "Direction-aware aggregation", fontsize=13.5, weight="bold")
    text(ax, 66.4, 21.1, "(schematic)", fontsize=12.5)

    ox, oy = 59.7, 11.7

    # axes
    arrow(ax, (ox, oy), (ox, 18.9), color=NAVY, linewidth=1.3, mutation_scale=11)
    arrow(ax, (ox, oy), (74.0, oy), color=NAVY, linewidth=1.3, mutation_scale=11)

    text(ax, 54.0, 16.1, "Sensitive\ndirection", fontsize=12.2, weight="bold", ha="center")
    text(ax, 54.0, 13.5, "(high curvature)", fontsize=10.8, ha="center")

    text(ax, 68.0, 9.4, "Flat direction", fontsize=12.2, weight="bold", ha="center")
    text(ax, 68.0, 7.8, "(low curvature)", fontsize=10.8, ha="center")

    # same magnitude arrows visually represented
    arrow(
        ax,
        (ox, oy),
        (62.0, 17.8),
        color=RED,
        linewidth=2.7,
        mutation_scale=15,
    )
    text(ax, 62.7, 17.1, "Stronger penalty", fontsize=11.0, weight="bold", color=RED)
    text(ax, 62.7, 15.7, "(same perturbation)", fontsize=10.0, weight="bold", color=RED)

    arrow(
        ax,
        (ox, oy),
        (66.4, 12.9),
        color=DIRECTION_BLUE,
        linewidth=2.7,
        mutation_scale=15,
    )
    text(ax, 67.1, 13.4, "Weaker penalty", fontsize=11.0, weight="bold", color=DIRECTION_BLUE)
    text(ax, 67.1, 12.0, "(same perturbation)", fontsize=10.0, weight="bold", color=DIRECTION_BLUE)

    # explanatory text box
    rounded_box(
        ax, 79.3, 7.9, 16.3, 8.7,
        facecolor="#F3EEFA",
        edgecolor="#D4C7E8",
        linewidth=0.9,
        radius=0.36,
        zorder=5,
    )
    text(
        ax,
        87.45,
        12.25,
        "The same magnitude of\nparameter change incurs\n"
        "different penalties along\ndifferent curvature directions.",
        fontsize=11.1,
        ha="center",
    )


def draw_broadcast(ax):
    # right vertical segment
    ax.plot(
        [87.8, 87.8],
        [7.0, 3.0],
        color=PURPLE,
        linewidth=2.7,
        solid_capstyle="round",
        zorder=6,
    )

    # long horizontal return
    ax.plot(
        [87.8, 21.4],
        [3.0, 3.0],
        color=PURPLE,
        linewidth=2.7,
        solid_capstyle="round",
        zorder=6,
    )

    # arrow back up into client panel
    arrow(
        ax,
        (21.4, 3.0),
        (21.4, 5.5),
        color=PURPLE,
        linewidth=2.7,
        mutation_scale=14,
    )

    text(
        ax,
        54.8,
        1.45,
        "Broadcast global experts to clients",
        fontsize=13.1,
        weight="bold",
        ha="center",
    )


# ============================================================
# Main
# ============================================================
def main():
    # Reference image is ~1805x871 => aspect ratio ~2.07
    fig, ax = plt.subplots(figsize=(18.05, 8.71))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48.5)
    ax.set_aspect("auto")
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_outer_panels(ax)
    draw_client_tabs(ax)
    draw_local_moe(ax)
    draw_tkfac(ax)
    draw_upload(ax)

    draw_server_header(ax)
    draw_weighted_initialization(ax)
    draw_fisher_optimization(ax)
    draw_global_experts(ax)
    draw_direction_aware(ax)
    draw_broadcast(ax)

    # Fixed margins: avoids bbox differences between PNG/PDF/SVG
    fig.subplots_adjust(left=0.012, right=0.992, top=0.985, bottom=0.025)

    png_path = OUTPUT_DIR / "curvfedmoe_reproduced.png"
    pdf_path = OUTPUT_DIR / "curvfedmoe_reproduced.pdf"
    svg_path = OUTPUT_DIR / "curvfedmoe_reproduced.svg"

    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(svg_path, facecolor="white")

    plt.close(fig)

    print("Finished.")
    print(f"PNG: {png_path.resolve()}")
    print(f"PDF: {pdf_path.resolve()}")
    print(f"SVG: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
