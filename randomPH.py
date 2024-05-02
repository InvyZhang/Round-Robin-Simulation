import numpy as np

class PHDistribution:
    def __init__(self, states, high1, high2, seed=None, cv_target='wide'):
        self.states = states
        if seed is not None:
            np.random.seed(seed)
        self.high1 = high1
        self.high2 = high2
        self.alpha = self.generate_alpha()
        self.T, self.t = self.generate_T_t()
        self.mu = None
        self.sigma = None
        self.CV = None
        self.calculate_metrics()
    
    def generate_alpha(self):
        # 生成归一化的初始概率向量 alpha
        alpha = np.random.dirichlet(np.ones(self.states))
        return np.array(alpha)

    def generate_T_t(self):
        # 根据 CV 目标生成转移率矩阵 T 和吸收率向量 t
        T = np.zeros((self.states, self.states))
        # 设置非对角线元素，即非吸收状态之间的转移率，使之较小以避免T退化
        for i in range(self.states):
            for j in range(self.states):
                if i != j:
                    T[i, j] = np.random.uniform(low=0, high=self.high1)
        t = np.random.uniform(low=0, high=self.high2, size=(self.states, 1))
        # 更新矩阵 T 的对角线元素使 T 行可逆
        for i in range(self.states):
            T[i, i] = -np.sum(T[i, :]) - t[i]  # 设置对角线元素
        
        return T, t

    def calculate_metrics(self):
        # 计算 PH 分布的均值和变异系数
        T_inv = np.linalg.inv(self.T)
        mu = -self.alpha.dot(T_inv.dot(np.ones(self.states)))
        sigma_squared = 2 * self.alpha.dot(T_inv).dot(T_inv).dot(np.ones(self.states)) - mu**2
        sigma = np.sqrt(sigma_squared)
        CV = sigma / mu if mu != 0 else 0
        
        self.mu = mu
        self.sigma = sigma
        self.CV = CV

    def __repr__(self):
        # 打印 PH 分布的参数和统计量
        return f"PH Distribution: alpha={self.alpha}, T=\n{self.T}\nt=\n{self.t}\nMean (mu): {self.mu}, Standard deviation (sigma): {self.sigma}, Coefficient of Variation (CV): {self.CV}\n" 

if __name__ == "__main__":
    ph_dist = PHDistribution(states = 8, high1 = 0.02, high2 = 0.1)
    print(ph_dist.alpha, ph_dist.T, ph_dist.t)
    print(ph_dist)