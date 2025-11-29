#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


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


# ===================== Ap（ 分母保护） =====================
def Ap(M, gammath, kai11, V11, lam1, HI, Xi, p, a):
    if len(a) < p:
        raise IndexError(f"Index p={p} out of bounds for power vector of length {len(a)}")

    if p == 1:
        num = gammath * kai11 * V11 * lam1 * (1 + HI) + gammath
        den_core = a[0] - gammath * (np.sum(a[1:]) + HI)
    else:
        sum_before = np.sum(a[:p - 1])
        sum_after = np.sum(a[p:]) if p < len(a) else 0.0
        num = gammath * kai11 * V11 * lam1 * (Xi * sum_before + sum_after + a[p - 1] + HI) + gammath
        den_core = a[p - 1] - gammath * (Xi * sum_before + sum_after + HI)

    # 分母必须 >0，否则视为极差/不可行
    if den_core <= 0:
        return 1e6

    den = kai11 * lam1 * den_core
    return num / den


# ===================== 单用户中断概率 P_p（原式 + 裁剪） =====================
def P(alphaSD1, mSD1, BSD1, p, a, Ap_value, deltaSD1):
    user_OP_sum = 0.0
    for k in range(int(mSD1)):
        theta = Theta(k, deltaSD1, mSD1)

        # 防止 exp 溢出
        x = -Ap_value * BSD1
        x = np.clip(x, -700, 700)
        exp_term = np.exp(x)

        factorial_k = math.factorial(k)
        first_term = factorial_k / (BSD1 ** (k + 1))

        second_term_sum = 0.0
        for n in range(k + 1):
            second_term_sum += (factorial_k / math.factorial(n)) * (Ap_value ** n) / (BSD1 ** (k - n + 1))

        user_OP = theta * (first_term - exp_term * second_term_sum)
        user_OP_sum += user_OP

    Pp = alphaSD1 * user_OP_sum

    # 概率裁剪到 [0,1]
    if np.isnan(Pp) or np.isinf(Pp):
        Pp = 1.0
    else:
        Pp = max(0.0, min(1.0, Pp))

    return Pp


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

        Pp = max(0.0, min(1.0, Pp))
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

            # Dirichlet 采样 + 降序
            particles[i, :] = np.sort(np.random.dirichlet(np.ones(dimensions)))[::-1]

            # 首用户约束
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


# ===================== “改进 PSO”（线性递减惯性） =====================
def pso_improved(fitness_func, bounds, num_particles, max_iter, *args):
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

    w_start, w_end = 0.9, 0.3
    c1 = c2 = 2.3
    v_max = 0.2
    for t in range(max_iter):
        w = w_end + (w_start - w_end) * np.exp(-7 * t / max_iter)

        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - particles[i])
                + c2 * r2 * (gbest_position - particles[i])
            )
            particles[i] += velocities[i]

            # 每步硬约束投影回可行域
            particles[i] = np.clip(particles[i], 1e-6, 1.0)

            if t % 8 == 0:
                particles[i:i + 1] = apply_constraints(
                    particles[i:i + 1], dim, args[9], args[8], args[10]
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

def de_opt(fitness_func, bounds, pop_size, max_gen, *args):
    """
    Differential Evolution (DE/rand/1/bin) with hard constraints via apply_constraints.
    Returns: best_value, best_curve
    """
    dim = len(bounds)

    # --- 1) 初始化：Dirichlet + 硬约束投影（与 PSO 等价的可行初始化） ---
    pop = np.random.dirichlet(np.ones(dim), size=pop_size)
    pop = np.sort(pop, axis=1)[:, ::-1]
    pop = apply_constraints(pop, dim, args[9], args[8], args[10])

    fit = np.array([fitness_func(ind, *args)[0] for ind in pop])

    best_idx = np.argmin(fit)
    best_val = fit[best_idx]
    best_curve = []

    # DE 参数（常用默认）
    F = 0.5   # mutation factor
    CR = 0.9  # crossover prob

    for g in range(max_gen):
        for i in range(pop_size):
            # --- 2) 变异：选 3 个不同的个体 ---
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            v = pop[r1] + F * (pop[r2] - pop[r3])

            # --- 3) 交叉（二项式交叉） ---
            j_rand = np.random.randint(dim)
            u = pop[i].copy()
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            # --- 4) 硬约束投影：保持可行域一致 ---
            u = np.clip(u, 1e-6, 1.0)
            u = u / np.sum(u)
            u = np.sort(u)[::-1]
            u = apply_constraints(u.reshape(1, -1), dim, args[9], args[8], args[10])[0]

            fu = fitness_func(u, *args)[0]

            # --- 5) 选择 ---
            if fu < fit[i]:
                pop[i] = u
                fit[i] = fu
                if fu < best_val:
                    best_val = fu
                    best_idx = i

        best_curve.append(best_val)

    return best_val, np.array(best_curve)


# ===================== 主程序：对比（M=6/9/12） =====================
if __name__ == "__main__":

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
    lam1_dB = 35  # 固定 SNR

    # 需要比较的 M
    M_list = [6, 9, 12]

    # PSO 迭代参数
    num_particles = 50
    max_iter = 100
    iteration = np.arange(1, max_iter + 1)

    # 保存 mat
    results_mat = {"iteration": iteration}

    # 画图
    plt.figure(figsize=(10, 6))

    for M in M_list:
        print(f"\n[INFO] Running for M={M}, SNR={lam1_dB} dB")

        init_args = initialize_parameters(
            mSD1, bSD1, omegaSD1,
            kai11, V11, ks, kD1,
            Xi, lam1_dB, M, eta
        )

        args = (
            int(init_args[5]),    # M
            float(init_args[1]),  # alphaSD1
            int(init_args[0]),    # mSD1
            float(init_args[2]),  # BSD1
            float(init_args[3]),  # lam1
            float(init_args[4]),  # deltaSD1
            float(init_args[6]),  # kai11
            float(init_args[7]),  # V11
            float(init_args[8]),  # HI
            float(init_args[9]),  # Xi
            float(init_args[10])  # eta
        )

        bounds = np.array([(0, 1)] * M)

        val_im, curve_im = pso_improved(fitness_func, bounds, num_particles, max_iter, *args)
        val_va, curve_va = pso_vanilla (fitness_func, bounds, num_particles, max_iter, *args)
        val_de, curve_de = de_opt(fitness_func, bounds, num_particles, max_iter, *args)

        print(f"  Improved PSO Final OP: {val_im:.4e}")
        print(f"  Vanilla  PSO Final OP: {val_va:.4e}")
        print(f"  De Final OP: {val_de:.4e}")

        # 画两条线
        plt.semilogy(iteration, curve_im, linestyle='-',  linewidth=2,
                     label=f'Improved PSO (M={M})')
        plt.semilogy(iteration, curve_va, linestyle='--', linewidth=2,
                     label=f'Vanilla PSO  (M={M})')
        plt.semilogy(iteration, curve_de, linestyle=':', linewidth=2,
                     label=f'DE (M={M})')

        # 存 mat
        results_mat[f'improved_M{M}'] = curve_im
        results_mat[f'vanilla_M{M}']  = curve_va
        results_mat[f'de_M{M}'] = curve_de

    plt.xlabel("Iteration")
    plt.ylabel("System Outage Probability (OP)")
    plt.title(r"Convergence: Improved PSO vs Vanilla PSO (SNR=35 dB)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(ncol=2, fontsize=10)
    plt.tight_layout()
    plt.show()

    savemat("improved_vs_vanilla_M6_9_12.mat", results_mat)
    print("\n✅ Data saved to improved_vs_vanilla_M6_9_12.mat")
