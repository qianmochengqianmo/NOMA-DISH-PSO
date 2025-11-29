#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


# ===================== 参数初始化 =====================
def initialize_parameters(mSD1, bSD1, omegaSD1, kai11, V11, ks, kD1, Xi, lam1_dB, M, eta):
    alphaSD1 = ((2 * bSD1 * mSD1) / (2 * bSD1 * mSD1 + omegaSD1)) ** mSD1 / (2 * bSD1)
    betaSD1 = 1 / (2 * bSD1)
    deltaSD1 = omegaSD1 / (2 * bSD1 * (2 * bSD1 * mSD1 + omegaSD1))
    BSD1 = betaSD1 - deltaSD1
    HI = ks ** 2 + kD1 ** 2
    lam1 = 10 ** (lam1_dB / 10)
    return mSD1, alphaSD1, BSD1, lam1, deltaSD1, M, kai11, V11, HI, Xi, eta


# ===================== 数学辅助函数 =====================
def Pochhammer(s, k):
    result = 1
    for i in range(k):
        result *= (s + i)
    return result


def Theta(k, deltaSD1, mSD1):
    if k == 0:
        return 1
    return ((-1) ** k * Pochhammer(1 - mSD1, k) * deltaSD1 ** k) / (math.factorial(k) ** 2)


# ===================== Ap（分母保护） =====================
def Ap(M, gammath, kai11, V11, lam1, HI, Xi, p, a):
    if p == 1:
        num = gammath * kai11 * V11 * lam1 * (1 + HI) + gammath
        den_core = a[0] - gammath * (np.sum(a[1:]) + HI)
    else:
        sum_before = np.sum(a[:p - 1])
        sum_after = np.sum(a[p:]) if p < len(a) else 0.0
        num = gammath * kai11 * V11 * lam1 * (Xi * sum_before + sum_after + a[p - 1] + HI) + gammath
        den_core = a[p - 1] - gammath * (Xi * sum_before + sum_after + HI)

    if den_core <= 0:
        return 1e6

    den = kai11 * lam1 * den_core
    return num / den


# ===================== 单用户中断概率 =====================
def P(alphaSD1, mSD1, BSD1, p, a, Ap_value, deltaSD1):
    user_OP_sum = 0.0
    for k in range(int(mSD1)):
        theta = Theta(k, deltaSD1, mSD1)

        x = -Ap_value * BSD1
        x = np.clip(x, -700, 700)
        exp_term = np.exp(x)

        factorial_k = math.factorial(k)
        first_term = factorial_k / (BSD1 ** (k + 1))

        second_term_sum = 0.0
        for n in range(k + 1):
            second_term_sum += (factorial_k / math.factorial(n)) * (Ap_value ** n) / (BSD1 ** (k - n + 1))

        user_OP_sum += theta * (first_term - exp_term * second_term_sum)

    Pp = alphaSD1 * user_OP_sum
    if np.isnan(Pp) or np.isinf(Pp):
        Pp = 1.0
    return max(0.0, min(1.0, Pp))


def gammath_for_user(p, eta):
    return eta ** (p - 1)


# ===================== 系统中断概率（fitness） =====================
def fitness_func(a, *args):
    M, alphaSD1, mSD1, BSD1, lam1, deltaSD1, kai11, V11, HI, Xi, eta = args
    total_OP = 1.0
    for p in range(1, M + 1):
        gammath_p = gammath_for_user(p, eta)
        Ap_value = Ap(M, gammath_p, kai11, V11, lam1, HI, Xi, p, a)
        Pp = P(alphaSD1, mSD1, BSD1, p, a, Ap_value, deltaSD1)
        total_OP *= (1 - Pp)
        if total_OP < 1e-12:
            total_OP = 0.0
            break
    outage = 1 - total_OP
    if np.isnan(outage) or np.isinf(outage):
        outage = 1.0
    return outage, None


# ===================== 硬约束投影（Dirichlet + SIC 检查） =====================
def apply_constraints(particles, dimensions, Xi, HI, eta, max_attempts=300):
    particles = np.array(particles)
    gammath1 = gammath_for_user(1, eta)

    for i in range(particles.shape[0]):
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            particles[i, :] = np.sort(np.random.dirichlet(np.ones(dimensions)))[::-1]

            if particles[i, 0] <= (np.sum(particles[i, 1:]) + HI) * gammath1:
                continue

            valid = True
            for p in range(1, dimensions + 1):
                sum_before = np.sum(particles[i, :p - 1]) if p > 1 else 0.0
                sum_after = np.sum(particles[i, p:]) if p < dimensions else 0.0
                rhs = (Xi * sum_before + sum_after + HI) * gammath_for_user(p, eta)
                if particles[i, p - 1] <= rhs:
                    valid = False
                    break

            if valid:
                break

    return particles


# ===================== Improved PSO（指数惯性，可调 ee） =====================
def pso_improved_exp(fitness_func, bounds, num_particles, max_iter, ee, *args):
    dim = len(bounds)

    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    particles = apply_constraints(particles, dim, args[9], args[8], args[10])

    pbest_positions = particles.copy()
    pbest_values = np.array([fitness_func(p, *args)[0] for p in particles])
    gbest_idx = np.argmin(pbest_values)
    gbest_position = particles[gbest_idx].copy()
    gbest_value = pbest_values[gbest_idx]

    gbest_curve = []

    w_start, w_end = 0.9, 0.3
    c1 = c2 = 2.3

    for t in range(max_iter):
        w = w_end + (w_start - w_end) * np.exp(-ee * t / max_iter)

        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - particles[i])
                + c2 * r2 * (gbest_position - particles[i])
            )

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 1e-6, 1.0)

            if t % 8 == 0:
                particles[i:i+1] = apply_constraints(
                    particles[i:i+1], dim, args[9], args[8], args[10]
                )

            fit = fitness_func(particles[i], *args)[0]
            if fit < pbest_values[i]:
                pbest_values[i] = fit
                pbest_positions[i] = particles[i].copy()
                if fit < gbest_value:
                    gbest_value = fit
                    gbest_position = particles[i].copy()

        gbest_curve.append(gbest_value)

    return gbest_value, np.array(gbest_curve)


# ===================== 传统 PSO（固定惯性 w=0.7） =====================
def pso_vanilla(fitness_func, bounds, num_particles, max_iter, *args):
    dim = len(bounds)

    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # 初次硬约束投影
    particles = apply_constraints(particles, dim, args[9], args[8], args[10])

    pbest_positions = particles.copy()
    pbest_values = np.array([fitness_func(p, *args)[0] for p in particles])

    gbest_idx = np.argmin(pbest_values)
    gbest_position = particles[gbest_idx].copy()
    gbest_value = pbest_values[gbest_idx]

    gbest_curve = []

    w = 0.7
    c1 = c2 = 2.3

    for t in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - particles[i])
                + c2 * r2 * (gbest_position - particles[i])
            )
            particles[i] += velocities[i]

            particles[i] = np.clip(particles[i], 1e-6, 1.0)
            particles[i:i+1] = apply_constraints(
                particles[i:i+1], dim, args[9], args[8], args[10]
            )

            fit = fitness_func(particles[i], *args)[0]

            if fit < pbest_values[i]:
                pbest_positions[i] = particles[i].copy()
                pbest_values[i] = fit

                if fit < gbest_value:
                    gbest_value = fit
                    gbest_position = particles[i].copy()

        gbest_curve.append(gbest_value)

    return gbest_value, np.array(gbest_curve)


# ===================== 主程序：SNR 扫描，ee=7，M=9 =====================
# ===================== 主程序：SNR 扫描，ee=7，M=9 =====================
if __name__ == "__main__":

    from scipy.io import savemat

    # 全局随机种子（保证整体复现）
    np.random.seed(42)

    # ===== 固定系统公共参数 =====
    kai11 = 0.5
    V11 = 0.001
    ks = 0.01
    kD1 = ks
    Xi = 0.01
    eta = 0.6

    M = 9            # 用户数
    ee = 7           # 指数惯性因子
    num_particles = 50
    max_iter = 100
    num_runs_pso = 3  # 每个 SNR 下 PSO 多次运行做平均，减小波动

    bounds = np.array([(0, 1)] * M)

    # ===== 信道参数三组组合 =====
    fading_profiles = [
        {"label": "Set1", "mSD1": 10, "bSD1": 0.158, "omegaSD1": 1.29},
        {"label": "Set2", "mSD1": 5,  "bSD1": 0.251, "omegaSD1": 0.278},
        {"label": "Set3", "mSD1": 1,  "bSD1": 0.063, "omegaSD1": 0.0007},
    ]

    # ===== SNR 横坐标 =====
    snr_dB_list = np.arange(5, 46, 5)   # 5,10,...,45 dB
    n_snr = len(snr_dB_list)
    n_sets = len(fading_profiles)

    # ===== 固定功率分配向量 a_fixed（降序，总和为 1）=====
    a_fixed = np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02])
    assert len(a_fixed) == M, "a_fixed 长度必须等于 M"
    a_fixed = a_fixed / np.sum(a_fixed)
    a_fixed = np.sort(a_fixed)[::-1]

    # ===== 先把 a_fixed 投影到可行域，一次性得到全局固定 baseline =====
    HI_global = ks**2 + kD1**2
    a_fixed_feasible = apply_constraints(
        a_fixed.reshape(1, -1),
        M,
        Xi,
        HI_global,
        eta
    )[0]  # 取出一行
    # 之后所有 SNR / 所有 fading set 都用这个 a_fixed_feasible

    # 用于存储结果：形状 [n_sets, n_snr]
    pso_OP_improved = np.zeros((n_sets, n_snr))
    pso_OP_vanilla  = np.zeros((n_sets, n_snr))
    fixed_OP        = np.zeros((n_sets, n_snr))

    # ===== 主循环：对每组 fading + 每个 SNR 计算 OP =====
    for i, prof in enumerate(fading_profiles):
        mSD1 = prof["mSD1"]
        bSD1 = prof["bSD1"]
        omegaSD1 = prof["omegaSD1"]

        print(f"=== {prof['label']} : m={mSD1}, b={bSD1}, omega={omegaSD1} ===")

        for j, lam1_dB in enumerate(snr_dB_list):
            # 构造该 (fading, SNR) 下的参数
            init_args = initialize_parameters(
                mSD1, bSD1, omegaSD1,
                kai11, V11, ks, kD1, Xi,
                lam1_dB, M, eta
            )
            # init_args: mSD1, alphaSD1, BSD1, lam1, deltaSD1, M, kai11, V11, HI, Xi, eta
            args = (
                int(init_args[5]),   # M
                float(init_args[1]), # alphaSD1
                int(init_args[0]),   # mSD1
                float(init_args[2]), # BSD1
                float(init_args[3]), # lam1
                float(init_args[4]), # deltaSD1
                float(init_args[6]), # kai11
                float(init_args[7]), # V11
                float(init_args[8]), # HI
                float(init_args[9]), # Xi
                float(init_args[10]) # eta
            )

            # ---- Improved PSO：多次运行，取平均 ----
            vals_im = []
            for r in range(num_runs_pso):
                # 为了可复现又有差异，每个 (set, snr, run) 用不同 seed
                np.random.seed(1000 + i*1000 + j*10 + r)
                val_im, _ = pso_improved_exp(
                    fitness_func,
                    bounds,
                    num_particles,
                    max_iter,
                    ee,
                    *args
                )
                vals_im.append(val_im)
            pso_OP_improved[i, j] = np.mean(vals_im)

            # ---- Vanilla PSO：多次运行，取平均 ----
            vals_va = []
            for r in range(num_runs_pso):
                np.random.seed(2000 + i*1000 + j*10 + r)
                val_va, _ = pso_vanilla(
                    fitness_func,
                    bounds,
                    num_particles,
                    max_iter,
                    *args
                )
                vals_va.append(val_va)
            pso_OP_vanilla[i, j] = np.mean(vals_va)

            # ---- 固定功率 a_fixed_feasible 的 OP（完全确定，不再随机）----
            val_fixed, _ = fitness_func(a_fixed_feasible, *args)
            fixed_OP[i, j] = val_fixed

            print(
                f"  SNR = {lam1_dB:2d} dB | "
                f"OP_Improved = {pso_OP_improved[i,j]:.3e} | "
                f"OP_Vanilla = {pso_OP_vanilla[i,j]:.3e} | "
                f"OP_Fixed = {val_fixed:.3e}"
            )

    # ===== 画图：SNR – OP 曲线 =====
    plt.figure(figsize=(9, 6))

    for i, prof in enumerate(fading_profiles):
        label_im = f"{prof['label']} - Improved PSO (ee=7)"
        label_va = f"{prof['label']} - Vanilla PSO"
        label_fx = f"{prof['label']} - Fixed a (projected)"

        plt.semilogy(
            snr_dB_list, pso_OP_improved[i, :],
            marker='o', linewidth=2,
            label=label_im
        )
        plt.semilogy(
            snr_dB_list, pso_OP_vanilla[i, :],
            marker='^', linestyle='-.', linewidth=2,
            label=label_va
        )
        plt.semilogy(
            snr_dB_list, fixed_OP[i, :],
            marker='s', linestyle='--', linewidth=2,
            label=label_fx
        )

    plt.xlabel("Average SNR (dB)")
    plt.ylabel("System Outage Probability (OP)")
    plt.title(r"OP vs SNR, $M=9$, $ee=7$ (Improved PSO vs Vanilla PSO vs Fixed Power)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout()

    plt.savefig("pso_vs_snr_M9_ee7_with_vanilla_smooth.png", dpi=300)
    plt.show()

    # ===== 保存 .mat 结果 =====
    fading_labels = np.array([prof["label"] for prof in fading_profiles], dtype=object)

    savemat("pso_snr_results_M9_ee7_with_vanilla_smooth.mat", {
        "snr_dB_list": snr_dB_list,
        "pso_OP_improved": pso_OP_improved,   # [n_sets x n_snr]
        "pso_OP_vanilla": pso_OP_vanilla,     # [n_sets x n_snr]
        "fixed_OP": fixed_OP,                 # [n_sets x n_snr]
        "fading_labels": fading_labels,
        "M": M,
        "ee": ee,
        "a_fixed": a_fixed,
        "a_fixed_feasible": a_fixed_feasible
    })

    print("Figure saved as 'pso_vs_snr_M9_ee7_with_vanilla_smooth.png'")
    print("MAT file saved as 'pso_snr_results_M9_ee7_with_vanilla_smooth.mat'")

