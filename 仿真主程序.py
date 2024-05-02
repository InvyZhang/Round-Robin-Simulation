from roundRobinPH import ArrivalService
from randomPH import PHDistribution
import time
import numpy as np
import pandas as pd
columns = [
    'n', 'service_rate', 'm', 'arrival_rate', 'CV_list', 'C_list', 
    'ksai', 'avg_wait_time', 'queue_waiting_length_list', 'avg_pkg_loss_rate'
]
df = pd.DataFrame(columns = columns)
for p in range(50000):
    print('No.{} circulation starts......'.format(p+1))
    n = np.random.randint(1, 11) + 10
    extrance_params = []
    entrance_buffer = []
    for i in range(n):
        ph_dist = PHDistribution(states = np.random.randint(1, 11), high1 = 0.02, high2 = 0.1)
        params = [ph_dist.alpha, ph_dist.T, ph_dist.t]
        extrance_params.append(['PH', params])
        entrance_buffer.append(np.random.randint(2, 21))
    m = np.random.randint(1, 11)
    ph_dist_service = PHDistribution(states = np.random.randint(1, 11), high1 = 0.02, high2 = 0.1)
    exit_params = ['PH', [ph_dist_service.alpha, ph_dist_service.T, ph_dist_service.t]]
    k = [np.random.randint(1, 6)]
    kwargs = {'lam{}'.format(i+1): extrance_params[i] for i in range(len(extrance_params))}
    metrics_1 = []
    metrics_2 = []
    metrics_3 = []
    for _ in range(2):
        start_time = time.time()
        print('No.{} sampling from No.{} circulation starts......'.format(_+1, p+1))
        ins = ArrivalService(
        n, 
        m, 
        **kwargs,
        u = exit_params,
        c = entrance_buffer,
        k = k
        )
        try:
            ins.start()
            accu_wait_time, serve_num = 0, 0
            for i in range(n):
                for j in range(len(ins.clients[i].server_time)):
                    serve_num += 1
                    accu_wait_time += ins.clients[i].out_time[j] - ins.clients[i].in_time[j] + ins.clients[i].server_time[j]
            avg_wait_time = accu_wait_time / serve_num
            queue_waiting_length_list = []
            for i in range(n):
                queue_waiting_length = 0
                for j in range(len(ins.clients[i].wait_time)):
                    queue_waiting_length += ins.clients[i].wait_time[j] * (j+1)
                queue_waiting_length_list.append(queue_waiting_length / ins.MAX_WORK_TIME)
            total, lost = 0, 0
            for i in range(n):
                total += ins.clients[i].total
                lost += ins.clients[i].lost
            avg_pkg_loss_rate = lost / total
            metrics_1.append(avg_wait_time)
            metrics_2.append(queue_waiting_length_list)
            metrics_3.append(avg_pkg_loss_rate)
        except:
            pass
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 300:
            print('Sampling took too long. Moving to the next circulation.')
            break
    avg_wait_time = np.mean(metrics_1)
    queue_waiting_length_list = [sum(sub_list) / len(metrics_2) for sub_list in zip(*metrics_2)]
    avg_pkg_loss_rate = np.mean(metrics_3)
    ins = ArrivalService(
        n, 
        m, 
        **kwargs,
        u = exit_params,
        c = entrance_buffer,
        k = k
    )
    data = {
        'n': ins.n,
        'service_rate': ins.service_rate(),
        'm': ins.m,
        'arrival_rate': ins.arrival_rate(),
        'CV_list': ins.coef_var(),
        'C_list': ins.lam_c,
        'ksai': ins.k,
        'avg_wait_time': avg_wait_time,
        'queue_waiting_length_list': queue_waiting_length_list,
        'avg_pkg_loss_rate': avg_pkg_loss_rate
    }
    df.loc[df.index.max() + 1 if not df.empty else 0] = data
    df.to_excel('round-robin simulation data_highCal.xlsx')