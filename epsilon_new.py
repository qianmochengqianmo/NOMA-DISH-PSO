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


# ===================== Vanilla PSO（固定 w） =====================
def pso_vanilla(fitness_func, bounds, num_particles, max_iter, *args):
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
                pbest_values[i] = fit
                pbest_positions[i] = particles[i].copy()
                if fit < gbest_value:
                    gbest_value = fit
                    gbest_position = particles[i].copy()

        gbest_curve.append(gbest_value)

    return gbest_value, np.array(gbest_curve)


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import savemat

    np.random.seed(42)

    # ===== 固定系统参数 =====
    mSD1 = 5
    bSD1 = 0.251
    omegaSD1 = 0.278

    kai11 = 0.5
    V11 = 0.001
    ks = 0.01
    kD1 = ks
    Xi = 0.01
    eta = 0.6
    lam1_dB = 35  # 平均SNR (dB)

    M = 10  # 用户数

    # ===== PSO 参数 =====
    num_particles = 50
    max_iter = 100
    bounds = np.array([(0, 1)] * M)
    iteration = np.arange(1, max_iter + 1)

    # ===== 构造 args（传给 fitness_func 和 PSO）=====
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

    # ===== ee 扫描列表 =====
    ee_list = [5, 6, 7, 8, 9]

    plt.figure(figsize=(10, 6))

    # ===== 1) 先跑 vanilla PSO =====
    print("Start vanilla PSO...")
    val_va, curve_va = pso_vanilla(
        fitness_func,
        bounds,
        num_particles,
        max_iter,
        *args
    )
    print(f"[Vanilla] Final OP = {val_va:.4e}")

    plt.semilogy(
        iteration,
        curve_va,
        linestyle='--',
        linewidth=2,
        label='Vanilla PSO'
    )

    # ===== 2) 再跑不同 ee 的 improved PSO =====
    print("Start improved PSO with exponential inertia...")
    improved_curves = []      # 存每个 ee 的收敛曲线
    improved_final_OP = []    # 存每个 ee 的最终 outage

    for ee in ee_list:
        print(f"  ee = {ee} ...")
        val_im, curve_im = pso_improved_exp(
            fitness_func,   # fitness function
            bounds,         # bounds
            num_particles,  # number of particles
            max_iter,       # iterations
            ee,             # 指数惯性因子
            *args           # 系统参数
        )

        improved_curves.append(curve_im)
        improved_final_OP.append(val_im)

        plt.semilogy(
            iteration,
            curve_im,
            linewidth=2,
            label=f'Improved PSO (ee={ee})'
        )

        print(f"[Improved ee={ee}] Final OP = {val_im:.4e}")

    # ===== 3) 画图设置 =====
    plt.xlabel("Iteration")
    plt.ylabel("System Outage Probability (OP)")
    plt.title(r"Effect of Exponential Inertia Factor $ee$ (M=9, SNR=35 dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, ncol=2)
    plt.tight_layout()

    # 保存为 PNG 图像
    plt.savefig("pso_ee_M9.png", dpi=300)
    plt.show()

    # ===== 4) 保存为 .mat 文件 =====
    improved_curves = np.array(improved_curves)      # 形状: [len(ee_list), max_iter]
    improved_final_OP = np.array(improved_final_OP)  # 形状: [len(ee_list)]

    savemat("pso_results_M9.mat", {
        "iteration": iteration,           # 1 x T
        "curve_va": curve_va,            # 1 x T
        "ee_list": np.array(ee_list),    # 1 x Ne
        "curves_improved": improved_curves,   # Ne x T
        "final_OP_vanilla": val_va,      # 标量
        "final_OP_improved": improved_final_OP  # 1 x Ne
    })

    print("Figure saved as 'pso_ee_M9.png'")
    print("MAT file saved as 'pso_results_M9.mat'")
