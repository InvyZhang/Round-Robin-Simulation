import random
import numpy as np

def generate_ph_random(alpha, T, t, size):

    result = []
    for _ in range(size):
        current_state = np.random.choice(len(alpha), p = alpha)
        time = 0
        
        while True:
            leave_rate = -T[current_state, current_state]
            time += np.random.exponential(1 / leave_rate)
            to_absorb_prob = t[current_state] / leave_rate
            
            if np.random.rand() < to_absorb_prob:
                break
            else:
                probs = T[current_state, :] / leave_rate
                probs[current_state] = 0
                probs = probs / probs.sum()
                current_state = np.random.choice(len(alpha), p = probs)
        result.append(time)
    
    return result

class ArrivalService:
    modes = {'Normal': 1, 'Exponential': 2, 'Uniform': 3, 'PH': 4}
    MAX_LIST_SIZE = 10
    MAX_WORK_TIME = 100000

    # 返回size大小的列表，满足给定的分布
    def calculate_time(self, mode, args):
        size = ArrivalService.MAX_LIST_SIZE
        assert mode in ArrivalService.modes, "Please input a right mode name"
        if mode == 'Normal':
            mean, std = args[0], args[1]
            return list(np.random.normal(mean, std, size))
        if mode == 'Exponential':
            lambd = args[0]
            return [random.expovariate(lambd) for _ in range(size)]
        if mode == 'Uniform':
            a, b = args[0], args[1]
            return list(np.random.uniform(a, b, size))
        if mode == 'PH':
            alpha, T, t = args[0], args[1], args[2]
            return generate_ph_random(alpha, T, t, size)

    # 定义客户端类
    class Client:
        def __init__(self, cid, mode, args, c, a_s):
            self.t = 0
            self.c = c
            self.id = cid
            self.mode = mode
            self.lost = 0
            self.length = 0
            self.total = 0  
            self.wait_time = []
            self.in_time = []
            self.out_time = []
            self.server_time = []
            for i in range(self.c):
                self.wait_time.append(0)
            assert self.mode in ArrivalService.modes, "Please input a right client mode"
            self.list = ArrivalService.calculate_time(a_s, mode, args)

    # 定义服务端类
    class Server:
        def __init__(self, sid, mode, args, a_s):
            self.t = 0
            self.id = sid
            self.mode = mode
            self.if_work = 0
            self.total = 0
            self.c_id = -1
            assert self.mode in ArrivalService.modes, "Please input a right server mode"
            self.list = ArrivalService.calculate_time(a_s, mode, args)

    # 输入参数类型为n,m,{[lam1.mode,lam1.pram...],...,[lamn.mode,lamn.pram...],[u.mode,u.pram...],[c1...cn],[k]}
    def __init__(self, *args, **kwargs):
        self.n = args[0]
        self.m = args[1]
        self.lam_mode = []
        self.lam_args = []
        count = 0
        for key, value in kwargs.items():
            if count < self.n:
                self.lam_mode.append(value[0])
                self.lam_args.append(value[1])
            elif count == self.n:
                self.servers = {i: self.Server(i, value[0], value[1], self) for i in range(self.m)}
                self.u_mode = value[0]
                self.u_args = value[1]
            elif count == self.n + 1:
                self.lam_c = value
            else:
                self.k = value[0]
            count = count + 1
        self.clients = {i: self.Client(i, self.lam_mode[i], self.lam_args[i], self.lam_c[i], self) for i in
                        range(self.n)}

    # 进行服务端和客户端的交互模拟
    def start(self):
        # 当前时间
        t = 0
        # 轮询顺序
        num = 0
        # 存储服务端的工作信息
        if_servers_work = []
        for i in range(self.m):
            if_servers_work.append(self.servers[i].if_work)

        while True:
            temp_client_t = []
            temp_server_t = []
            # 计算每个服务端的下一个请求的到达时间
            for i in range(self.n):
                if len(self.clients[i].list) == 0:
                    self.clients[i].list = self.calculate_time(self.lam_mode[i], self.lam_args[i])
                temp_client_t.append(self.clients[i].t + self.clients[i].list[0])

            # 计算每个客户端的下一个请求的处理的结束时间
            for i in range(self.m):
                if if_servers_work[i] == 0:
                    temp_server_t.append(self.MAX_WORK_TIME)
                    continue
                if len(self.servers[i].list) == 0:
                    self.servers[i].list = self.calculate_time(self.u_mode, self.u_args)
                temp_server_t.append(self.servers[i].t + self.servers[i].list[0])

            # 找到最小的时间以及索引
            min_client_value = min(temp_client_t)
            min_server_value = min(temp_server_t)
            client_index = temp_client_t.index(min_client_value)
            server_index = temp_server_t.index(min_server_value)

            # 客户端请求入队列
            if min_client_value < min_server_value:
                # 删除其第一个值，模拟时间推进
                del self.clients[client_index].list[0]
                pre_t = self.clients[client_index].t
                self.clients[client_index].t = min_client_value
                t = self.clients[client_index].t
                # 计算队列长度
                if self.clients[client_index].length != 0:
                    self.clients[client_index].wait_time[self.clients[client_index].length-1] = self.clients[client_index].wait_time[self.clients[client_index].length-1] + t - pre_t
                # 丢包情况
                if self.clients[client_index].length == self.clients[client_index].c:
                    self.clients[client_index].lost = self.clients[client_index].lost + 1
                    # print("当前时间为：{}，客户端{}发生丢包".format(t, self.clients[client_index].id))
                else:
                    self.clients[client_index].length = self.clients[client_index].length + 1
                    # print("当前时间为：{}，客户端{}请求成功入队".format(t, self.clients[client_index].id))
                    self.clients[client_index].in_time.append(t)
                self.clients[client_index].total = self.clients[client_index].total + 1

            # 服务端处理请求结束
            else:
                # 删除其第一个值，模拟时间推进
                self.clients[self.servers[server_index].c_id].server_time.append(self.servers[server_index].list[0])
                del self.servers[server_index].list[0]
                self.servers[server_index].t = min_server_value
                t = self.servers[server_index].t
                # print("当前时间为：{}，服务端{}处理请求结束".format(t, self.servers[server_index].id))
                self.servers[server_index].total = self.servers[server_index].total + 1
                # 释放客户端
                self.servers[server_index].if_work = 0
                if_servers_work[server_index] = 0

            # 生成请求队列，存储每个客户端队列的排队数量
            temp_list = []
            for i in range(self.n):
                temp_list.append(self.clients[i].length)
            # 模拟轮询，此时有空闲的服务端并且有请求的情况
            if max(temp_list) != 0:
                if min(if_servers_work) == 0:
                    s_index = if_servers_work.index(0)
                    c_index = self.poll(temp_list, num)
                    num = c_index
                    self.clients[c_index].length = self.clients[c_index].length - 1
                    self.servers[s_index].if_work = 1
                    self.servers[s_index].t = t
                    self.servers[s_index].c_id = c_index
                    if_servers_work[s_index] = 1
                    # print("当前时间为：{}，服务端{}开始处理来自客户端{}请求".format(t, s_index, c_index))
                    self.clients[c_index].out_time.append(t)

            # t等于最大运行时间结束进程
            if t >= self.MAX_WORK_TIME:
                break

    # 计算客户端请求的到达率：到达时间间隔均值的倒数
    def arrival_rate(self):
        temp_list = []
        for i in range(self.n):
            if self.lam_mode[i] == 'Normal':
                temp_list.append(1 / self.lam_args[i][0])
            if self.lam_mode[i] == 'Exponential':
                temp_list.append(self.lam_args[i][0])
            if self.lam_mode[i] == 'Uniform':
                temp_list.append(2 / (self.lam_args[i][0] + self.lam_args[i][1]))
            if self.lam_mode[i] == 'PH':
                temp_list.append(1 / (-self.lam_args[i][0].dot(np.linalg.inv(self.lam_args[i][1]).dot(np.ones(self.lam_args[i][1].shape[0])))))
        return temp_list

    # 计算服务端的服务率
    def service_rate(self):
        temp_list = []
        if self.u_mode == 'Normal':
            temp_list.append(1 / self.u_args[0])
        elif self.u_mode == 'Exponential':
            temp_list.append(self.u_args[0])
        elif self.u_mode == 'Uniform':
            temp_list.append(2 / (self.u_args[0] + self.u_args[1]))
        elif self.u_mode == 'PH':
            temp_list.append(1 / (-self.u_args[0].dot(np.linalg.inv(self.u_args[1]).dot(np.ones(self.u_args[1].shape[0])))))
        return temp_list[0]
    
    # 计算客户端请求时间间隔的变异系数
    def coef_var(self):
        temp_list = []
        for i in range(self.n):
            if self.lam_mode[i] == 'Normal':
                temp_list.append(self.lam_args[i][1] / self.lam_args[i][0])
            if self.lam_mode[i] == 'Exponential': # 指数分布的变异系数 = 均值
                temp_list.append(1 / self.lam_args[i][0])
            if self.lam_mode[i] == 'Uniform':
                temp_list.append(12**(0.5) * (self.lam_args[i][1] - self.lam_args[i][0])/(self.lam_args[i][1] + self.lam_args[i][0]))
            if self.lam_mode[i] == 'PH':
                temp_list.append(((np.sqrt(2 * self.lam_args[i][0] @ np.linalg.matrix_power(np.linalg.inv(self.lam_args[i][1]), 2) @ np.ones((len(self.lam_args[i][1]), 1)) - (self.lam_args[i][0] @ np.linalg.inv(self.lam_args[i][1]) @ np.ones((len(self.lam_args[i][1]), 1)))**2)) / (-self.lam_args[i][0] @ np.linalg.inv(self.lam_args[i][1]) @ np.ones((len(self.lam_args[i][1]), 1))))[0])
        return temp_list

    # 定义轮询顺序
    def poll(self, temp_list, num):
        temp_num = num
        if self.k == 1: # 顺序轮询
            for i in range(self.n):
                temp_num = (temp_num + 1) % self.n
                if temp_list[temp_num] != 0:
                    break
        elif self.k == 2: # 随机轮询
            temp_l = []
            for i in range(self.n):
                if temp_list[i] != 0:
                    temp_l.append(i)
            temp_num = random.choice(temp_l)
        elif self.k == 3: # 加权轮询-当前请求数量 最大
            temp_l = []
            for i in range(self.n):
                temp_l.append(temp_list[i])
            temp_num = temp_l.index(max(temp_l))
        elif self.k == 4: # 加权轮询-当前请求数量 / 队列容量 最大
            temp_l = []
            for i in range(self.n):
                temp_l.append(temp_list[i] / self.lam_c[i])
            temp_num = temp_l.index(max(temp_l))
        elif self.k == 5: # 加权轮询-当前请求数量 * 队列到达率 最大
            temp_l = []
            for i in range(self.n):
                temp_l.append(temp_list[i] * self.arrival_rate()[i])
            temp_num = temp_l.index(max(temp_l))
        return temp_num

    # 输出clients所有参数
    def show_clients(self):
        for i in range(len(self.clients)):
            print(vars(self.clients[i]))

    # 输出servers所有参数
    def show_servers(self):
        for i in range(len(self.servers)):
            print(vars(self.servers[i]))

if __name__ == '__main__':
    ins = ArrivalService(3, 1, 
                        lam1 = ['PH', [np.array([0.2, 0.8]), np.array([[-2, 1], [0, -3]]), np.array([[1], [3]])]], 
                        lam2 = ['Normal', [0.5, 0.1]], 
                        lam3 = ['Exponential', [2]], 
                        u = ['Normal', [0.5, 0.1]],
                        c = [10, 10, 15],
                        k = [4]
                    )
    '''
    上述参数的含义：·
    n = 2(客户端中存在两个队列), m = 1(服务端中存在一个服务器)
    lam1 = ... 表示客户端中第一个队列的到达率服从 N(0.5, 0.1^2)
    u = ... 表示服务端中服务器的服务率服从 N(0.5, 0.1^2)
    c = [5, 4] 表示客户端中两个队列的缓冲区容量为5和4
    k = [1] 表示使用第一种轮询策略
    '''
    print(vars(ins))
    ins.show_clients()
    ins.show_servers()
    ins.start()
    ins.show_clients()