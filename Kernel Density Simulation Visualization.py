# -*- coding: utf-8 -*-
# 依赖: numpy, matplotlib, scipy
# pip install numpy matplotlib scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap


# ---------- 公共工具 ----------
def alpha_cmap(base_name, n=14, vmin=0.25, vmax=1.0, amin=0.06, amax=0.88):
    """
    从现有 colormap 采样，同时让 alpha 从 amin->amax 递增，
    形成“外圈更透明、内圈更实”的柔和云团效果。
    """
    base = plt.get_cmap(base_name)(np.linspace(vmin, vmax, n))
    alphas = np.linspace(amin, amax, n)
    base[:, -1] = alphas
    return ListedColormap(base)

def make_kde_fields(X1, X2, xlim, ylim, grid_n=300):
    """
    给两类 2xN 数据做 KDE 并在网格上评估，返回网格和两类归一化密度场
    """
    kde1 = gaussian_kde(X1)
    kde2 = gaussian_kde(X2)

    xx, yy = np.mgrid[xlim[0]:xlim[1]:complex(grid_n),
                      ylim[0]:ylim[1]:complex(grid_n)]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    z1 = kde1(grid).reshape(xx.shape)
    z2 = kde2(grid).reshape(xx.shape)

    # 归一化到 [0,1]，便于统一 levels/透明度
    z1 = (z1 - z1.min()) / (z1.max() - z1.min())
    z2 = (z2 - z2.min()) / (z2.max() - z2.min())
    return xx, yy, z1, z2

def draw_overlay(xx, yy, zA, zB, cmapA, cmapB,
                 xlim, ylim, levels=None, outfile="out.png",
                 show_axes=False, dpi=300):
    """
    将两张密度图叠加，输出透明 PNG
    """
    if levels is None:
        levels = np.linspace(0.05, 1.0, 14)

    fig = plt.figure(figsize=(7.0, 5.6), dpi=220)
    ax = plt.gca()

    # 叠加两类密度（顺序可调整）
    ax.contourf(xx, yy, zA, levels=levels, cmap=cmapA, antialiased=True)
    ax.contourf(xx, yy, zB, levels=levels, cmap=cmapB, antialiased=True)

    # 坐标轴表现
    if show_axes:
        # 中心轴
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.set_xticks(np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) + 1, 1.0))
        ax.set_yticks(np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) + 1, 1.0))
    else:
        # 完全无轴
        for s in ["top", "right", "left", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, transparent=True)
    # 如需矢量图：
    # plt.savefig(outfile.replace(".png", ".svg"), transparent=True)
    plt.close(fig)


# =========================================================
# 图 1：蓝/红，两类（方差较小），无坐标轴
# =========================================================
rng = np.random.default_rng(123)

# 蓝色类（中心 0,0），较小方差
Xb = rng.multivariate_normal(
    mean=[0.0, 0.0],
    cov=[[0.5, 0.10],
         [0.10, 0.25]],
    size=1500
).T  # shape -> (2, N)

# 红色类（中心 3.0,-0.8），较小方差
Xr = rng.multivariate_normal(
    mean=[3.0, -0.8],
    cov=[[0.35, -0.05],
         [-0.05, 0.18]],
    size=1500
).T

xlim1 = (-2.0, 5.0)
ylim1 = (-2.0, 2.0)

xx1, yy1, zb, zr = make_kde_fields(Xb, Xr, xlim1, ylim1, grid_n=300)
cb = alpha_cmap("Blues")   # 蓝
cr = alpha_cmap("Reds")    # 红
draw_overlay(xx1, yy1, zb, zr, cb, cr, xlim1, ylim1,
             outfile="gradient_kde_compact.png",
             show_axes=False, dpi=300)


# =========================================================
# 图 2：绿/黄，两类（均值位置不同），无坐标轴
#    绿色类中心移到 (-0.8, 0.6)
#    黄色类中心移到 (2.6, -1.4)
#    方差同样较小，范围略作调整
# =========================================================
rng2 = np.random.default_rng(456)

Xg = rng2.multivariate_normal(
    mean=[-0.8, 0.6],
    cov=[[0.45, 0.06],
         [0.06, 0.22]],
    size=1500
).T

Xy = rng2.multivariate_normal(
    mean=[2.4, -0.8],
    cov=[[0.32, -0.04],
         [-0.04, 0.16]],
    size=1500
).T

xlim2 = (-2.5, 5.0)
ylim2 = (-2.2, 2.4)

xx2, yy2, zg, zy = make_kde_fields(Xg, Xy, xlim2, ylim2, grid_n=300)

# 绿色与黄色配色：Greens + Wistia（明亮黄）
cg = alpha_cmap("Greens")
cy = alpha_cmap("Wistia")

draw_overlay(xx2, yy2, zg, zy, cg, cy, xlim2, ylim2,
             outfile="gradient_kde_green_yellow.png",
             show_axes=False, dpi=300)

print("已生成：gradient_kde_compact.png 和 gradient_kde_green_yellow.png")
