#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


# ===================== 参数初始化（和你原来一致） =====================
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


# ===================== Ap（在你原式基础上加少量数值保护） =====================
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

    # 分母必须 >0，否则 Ap 在物理上就是“极差”：近似中断
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
    # 你的设定：γ_th,k = η^(k-1) * γ_th,1（γ_th,1 已吸收/归一化）
    return eta ** (p - 1)


# ===================== 适应度函数：系统中断概率（原逻辑 + 裁剪） =====================
def fitness_func(a, *args):
    M, alphaSD1, mSD1, BSD1, lam1, deltaSD1, kai11, V11, HI, Xi, eta = args
    total_OP = 1.0

    for p in range(1, M + 1):
        gammath_p = gammath_for_user(p, eta)
        Ap_value = Ap(M, gammath_p, kai11, V11, lam1, HI, Xi, p, a)
        Pp = P(alphaSD1, mSD1, BSD1, p, a, Ap_value, deltaSD1)

        Pp = max(0.0, min(1.0, Pp))
        total_OP *= (1 - Pp)

        # 已接近 0 就提前结束
        if total_OP < 1e-12:
            total_OP = 0.0
            break

    outage = 1 - total_OP
    if np.isnan(outage) or np.isinf(outage):
        outage = 1.0

    return outage, None


# ===================== 约束处理：沿用你原来思路 =====================
def apply_constraints(particles, dimensions, Xi, HI, eta, max_attempts=300):
    """
    和你原来的写法一致：Dirichlet 采样 + 降序 + 检查所有 SIC 约束。
    只是把 max_attempts 从 1000 稍微降到 300，防止卡死。
    """
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

        # 如果 max_attempts 内都没找到严格可行解，就保留最后一次（和你原来的逻辑接近）
    return particles


# ===================== PSO（核心逻辑保持不变） =====================
def pso(fitness_func, bounds, num_particles, max_iter, *args):
    dim = len(bounds)
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # 初次强制施加约束
    particles = apply_constraints(particles, dim, args[9], args[8], args[10])

    pbest_positions = particles.copy()
    pbest_values = [fitness_func(p, *args)[0] for p in particles]
    gbest_value = min(pbest_values)
    gbest_position = particles[np.argmin(pbest_values)].copy()

    w_start, w_end = 0.9, 0.3

    for t in range(max_iter):
        w = w_end + (w_start - w_end) * np.exp(-7 * t / max_iter)
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                w * velocities[i]
                + 2.05 * r1 * (pbest_positions[i] - particles[i])
                + 2.05 * r2 * (gbest_position - particles[i])
            )
            particles[i] += velocities[i]

            # 保持在 [0,1]，再重新施加约束
            particles[i] = np.clip(particles[i], 1e-6, 1.0)
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

    return gbest_value


# ===================== 主函数：横坐标 kai11，一张图 9 条线 =====================
if __name__ == "__main__":
    from scipy.io import savemat

    # ---------- 固定信道 / 系统参数 ----------
    mSD1 = 5
    bSD1 = 0.251
    omegaSD1 = 0.278

    # 用户数固定为 3（建议先验证后再扩展）
    M = 9

    # 固定 eta、ks、V11、kD1
    eta = 0.6
    ks_fixed = 0.01
    kD1_fixed = ks_fixed
    V11_fixed = 0.001

    # ========= 3 个固定的 SNR(dB) =========
    SNR_dB_list = [25, 35, 45]

    # ========= 3 个 Xi =========
    Xi_values = [0.01, 0.05, 0.1]

    # ========= 横坐标：kai11 从 0.01 到 0.10 =========
    kai11_grid = np.linspace(0.01, 0.10, 20)

    # PSO 参数
    num_particles = 25
    max_iter = 40
    bounds = np.array([(0, 1)] * M)

    # 保存数据
    export_data = {
        "kai11_grid": kai11_grid,
        "SNR_dB_list": np.array(SNR_dB_list),
        "Xi_values": np.array(Xi_values),
    }

    all_results = {}

    # =============== 计算 9 条曲线（3 SNR × 3 Xi） =================
    for Xi in Xi_values:
        for lam1_dB in SNR_dB_list:

            outages_vs_kai = []

            for kai11 in kai11_grid:
                init_args = initialize_parameters(
                    mSD1, bSD1, omegaSD1,
                    kai11, V11_fixed, ks_fixed, kD1_fixed,
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

                outage = pso(fitness_func, bounds, num_particles, max_iter, *args)
                outages_vs_kai.append(outage)

            label = f"SNR={lam1_dB}dB, Xi={Xi}"
            all_results[label] = np.array(outages_vs_kai)

            # 保存到 MAT
            key_name = f"SNR_{lam1_dB}_Xi_{str(Xi).replace('.', '_')}"
            export_data[key_name] = np.array(outages_vs_kai)

    # =============== 画一张图：9 条曲线 =================
    plt.figure(figsize=(9, 7))

    for label, outages in all_results.items():
        plt.plot(kai11_grid, outages, marker='o', label=label)

    plt.title("Outage Probability vs kai11 (M=3, ks=0.01, V11=0.001, eta=0.6)")
    plt.xlabel("kai11")
    plt.ylabel("Outage Probability")
    plt.grid(True)
    plt.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.show()

    # =============== 保存 MAT 文件 =================
    savemat("outage_vs_kai_9curves_SNR_Xi.mat", export_data)
