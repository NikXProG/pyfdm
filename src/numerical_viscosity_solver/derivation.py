from .config import ViscosityConfig


def derivation_summary(config: ViscosityConfig) -> str:
    lines = [
        "Модифицированное уравнение для upwind-схемы (a > 0):",
        "(u_i^{n+1} - u_i^n)/dt + a (u_i^n - u_{i-1}^n)/dx = 0",
        "",
        "После разложения Тейлора:",
        "u_t + a u_x = epsilon_num * u_xx + O(dx, dt)",
        "epsilon_num = (a * dx / 2) * (1 - mu), mu = a*dt/dx",
        "",
        "Подстановка текущих параметров:",
        f"a={config.a:.3f}, dx={config.dx:.3f}, mu={config.mu:.3f}, dt={config.dt:.4f}",
        f"epsilon_num = {config.epsilon_num:.6f}",
    ]
    return "\n".join(lines)
