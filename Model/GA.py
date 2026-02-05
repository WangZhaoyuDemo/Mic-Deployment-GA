import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import data_process
import util

df_node, df_mic, df_layer, df_callGraph = data_process.build_data_df(3)
mic, layer, node, call_graph = data_process.data_process(df_node, df_mic, df_layer, df_callGraph)
# 建立通信网络矩阵
bandwidth_matrix = util.build_bandwidth_matrix(node)
# 建立调用图邻接矩阵和所有连通分量
communicate_matrix, node_mapping = util.build_adjacency_matrix(call_graph)


def adjust(x, classify_mic, start, end, max_attempts=50):
    """
    调整个体，使其满足节点 CPU/MEM 约束
    """
    node_cpu = [i for i in df_node['CPU']]
    node_mem = [i for i in df_node['MEM']]

    for i in range(len(x)):
        node_cpu[x[i]] -= mic[classify_mic[i]]['CPU']
        node_mem[x[i]] -= mic[classify_mic[i]]['MEM']

        attempts = 0
        while (node_cpu[x[i]] < 0 or node_mem[x[i]] < 0) and attempts < max_attempts:
            node_cpu[x[i]] += mic[classify_mic[i]]['CPU']
            node_mem[x[i]] += mic[classify_mic[i]]['MEM']
            x[i] = random.randint(start, end - 1)
            node_cpu[x[i]] -= mic[classify_mic[i]]['CPU']
            node_mem[x[i]] -= mic[classify_mic[i]]['MEM']
            attempts += 1

        if attempts >= max_attempts:
            # 如果尝试次数用尽，随机分配一个合法节点（可能不完美）
            x[i] = random.randint(start, end - 1)

    return x


# 计算部署成本，通信成本和部署时间
def get_baseline(x, classify_mic):
    matrix_n = np.zeros((len(df_node['ID']), len(df_layer['ID'])))
    node_cpu = [i for i in df_node['CPU']]
    node_mem = [i for i in df_node['MEM']]
    deployment_cost = 0
    time_cost = np.zeros(len(df_node['ID']))
    communicate_cost = 0
    for i in range(len(x)):
        # print(node_cpu[x[i]], node_mem[x[i]])
        node_cpu[x[i]] -= mic[classify_mic[i]]['CPU']
        node_mem[x[i]] -= mic[classify_mic[i]]['MEM']
        if node_cpu[x[i]] < 0 or node_mem[x[i]] < 0:
            return 999999
        deployment_cost += mic[classify_mic[i]]['Cost']
        temp_layer_list = str(mic[classify_mic[i]]['Layer']).split(',')
        layer_array = np.array(temp_layer_list, dtype=int)
        time_cost[x[i]] += mic[classify_mic[i]]['Cost'] / node[x[i]]['Bandwidth']
        for temp_layer in layer_array:
            if matrix_n[x[i]][temp_layer] == 1:
                deployment_cost -= layer[temp_layer]['Size']
            matrix_n[x[i]][temp_layer] = 1
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            # print("i and j:", i, j)
            # print("classify i and j:", classify_mic[i], classify_mic[j])
            # print(node_mapping[mic[classify_mic[i]]['Microservice ID']],
            # node_mapping[mic[classify_mic[j]]['Microservice ID']])
            # print("i, j")
            # print(node_mapping[mic[classify_mic[i]]['Microservice ID']], mic[classify_mic[i]]['Microservice ID'])
            # print(node_mapping[mic[classify_mic[j]]['Microservice ID']], mic[classify_mic[j]]['Microservice ID'])
            if communicate_matrix[node_mapping[mic[classify_mic[i]]['Microservice ID']]][node_mapping[mic[classify_mic[j]]['Microservice ID']]] != 0:
                communicate_cost += communicate_matrix[node_mapping[mic[classify_mic[i]]['Microservice ID']]][node_mapping[mic[classify_mic[j]]['Microservice ID']]] / (mic[classify_mic[i]]['Number'] * mic[classify_mic[j]]['Number'] * bandwidth_matrix[x[i]][x[j]])
    result = [deployment_cost, np.max(time_cost), communicate_cost]
    return result


# 负载均衡部署
def generate_load_balance_solution(classify_mic):
    node_cpu = np.array(df_node['CPU'])
    node_mem = np.array(df_node['MEM'])
    d = []

    for mic_id in classify_mic:
        cpu_need = mic[mic_id]['CPU']
        mem_need = mic[mic_id]['MEM']

        # 计算每个节点剩余资源
        remain_cpu = node_cpu - cpu_need
        remain_mem = node_mem - mem_need

        # 剔除无法部署的节点
        valid = (remain_cpu >= 0) & (remain_mem >= 0)
        if not np.any(valid):
            return None  # 无法部署

        # 选择剩余资源最“空”的节点（CPU+MEM 最大）
        score = remain_cpu + remain_mem
        idx = np.argmax(score * valid)

        # 记录部署
        d.append(idx)
        node_cpu[idx] -= cpu_need
        node_mem[idx] -= mem_need

    return np.array(d)


# 集约部署
def generate_consolidation_solution(classify_mic):
    # 当前节点剩余资源
    node_cpu = np.array(df_node['CPU'])
    node_mem = np.array(df_node['MEM'])

    # 微服务排序（按 CPU+MEM 降序） → 真正的集约策略
    sorted_index = sorted(range(len(classify_mic)),
                          key=lambda i: mic[classify_mic[i]]['CPU'] + mic[classify_mic[i]]['MEM'],
                          reverse=True)

    # 结果数组
    d = np.zeros(len(classify_mic), dtype=int)

    used_nodes = []

    for idx in sorted_index:
        m = classify_mic[idx]
        cpu_need = mic[m]['CPU']
        mem_need = mic[m]['MEM']

        placed = False

        # 1. 尝试放入已启用节点
        for n in used_nodes:
            if node_cpu[n] >= cpu_need and node_mem[n] >= mem_need:
                d[idx] = n
                node_cpu[n] -= cpu_need
                node_mem[n] -= mem_need
                placed = True
                break

        if placed:
            continue

        # 2. 需要新的节点
        for new_node in range(len(node_cpu)):
            if new_node not in used_nodes:
                if node_cpu[new_node] >= cpu_need and node_mem[new_node] >= mem_need:
                    used_nodes.append(new_node)
                    d[idx] = new_node
                    node_cpu[new_node] -= cpu_need
                    node_mem[new_node] -= mem_need
                    placed = True
                    break

        if not placed:
            return None  # 无法部署

    return d


# 计算部署成本，通信成本和部署时间
def function(x, classify_mic, baseline):
    matrix_n = np.zeros((len(df_node['ID']), len(df_layer['ID'])))
    node_cpu = [i for i in df_node['CPU']]
    node_mem = [i for i in df_node['MEM']]
    deployment_cost = 0
    time_cost = np.zeros(len(df_node['ID']))
    communicate_cost = 0
    for i in range(len(x)):
        node_cpu[x[i]] -= mic[classify_mic[i]]['CPU']
        node_mem[x[i]] -= mic[classify_mic[i]]['MEM']
        if node_cpu[x[i]] < 0 or node_mem[x[i]] < 0:
            return 999999
        deployment_cost += mic[classify_mic[i]]['Cost']
        temp_layer_list = str(mic[classify_mic[i]]['Layer']).split(',')
        layer_array = np.array(temp_layer_list, dtype=int)
        time_cost[x[i]] += mic[classify_mic[i]]['Cost'] / node[x[i]]['Bandwidth']
        for temp_layer in layer_array:
            if matrix_n[x[i]][temp_layer] == 1:
                deployment_cost -= layer[temp_layer]['Size']
            matrix_n[x[i]][temp_layer] = 1
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            if communicate_matrix[node_mapping[mic[classify_mic[i]]['Microservice ID']]][node_mapping[mic[classify_mic[j]]['Microservice ID']]] != 0:
                communicate_cost += communicate_matrix[node_mapping[mic[classify_mic[i]]['Microservice ID']]][node_mapping[mic[classify_mic[j]]['Microservice ID']]] / (mic[classify_mic[i]]['Number'] * mic[classify_mic[j]]['Number'] * bandwidth_matrix[x[i]][x[j]])
    return deployment_cost/baseline[0] + np.max(time_cost)/baseline[1] + communicate_cost/baseline[2]
    # return deployment_cost/baseline[0]
    # return np.max(time_cost)/baseline[1]
    # return communicate_cost/baseline[2]


# # 选择函数，三个参数：种群个体、种群数量、适应度值
# def select(population, population_size, fitness):
#     # fitness_proportion用来保存每个个体的选择概率
#     fitness_proportion = []
#     # 计算适应度之和
#     fitness_sum = 0
#     for i in range(population_size):
#         fitness_sum += fitness[i]
#     # 计算每个个体的选择概率
#     for i in range(population_size):
#         fitness_proportion.append(fitness[i] / fitness_sum)
#     # pie_fitness保存每个个体的累计概率
#     pie_fitness = []
#     cumsum = 0.0
#     for i in range(population_size):
#         pie_fitness.append(cumsum + fitness_proportion[i])  # pie_fitness为由1到i个个体相加的适应度和组成的list
#         cumsum += fitness_proportion[i]  # 所有个体生存率之和
#     # 生成随机数在轮盘上选点[0, 1)
#     random_selection = []
#     for i in range(population_size):
#         random_selection.append(random.random())  # 返回随机生成的一个实数，它在[0,1)范围内。
#     # 选择新种群
#     new_population = []
#     random_selection_id = 0
#     while random_selection_id < population_size:
#         # 随机数处于个体对应的累计区间时，则将这个个体赋给新种群
#         for i in range(population_size):
#             if random_selection[random_selection_id] < pie_fitness[i]:
#                 new_population.append(population[i])
#                 break
#         random_selection_id += 1
#     population = new_population
#     return population


def select(population, population_size, fitness):
    # fitness_proportion用来保存每个个体的选择概率
    fitness_proportion = []

    # 计算适应度倒数并归一化
    inverse_fitness = [1 / (f + 1e-6) for f in fitness]  # 加1e-6以避免除以0
    inverse_fitness_sum = sum(inverse_fitness)

    # 计算每个个体的选择概率
    fitness_proportion = [f / inverse_fitness_sum for f in inverse_fitness]

    # pie_fitness保存每个个体的累计概率
    pie_fitness = []
    cumsum = 0.0
    for proportion in fitness_proportion:
        cumsum += proportion
        pie_fitness.append(cumsum)  # pie_fitness为由1到i个体相加的适应度和组成的list

    # 生成随机数在轮盘上选点[0, 1)
    random_selection = [random.random() for _ in range(population_size)]

    # 选择新种群
    new_population = []
    random_selection_id = 0
    while random_selection_id < population_size:
        rand = random_selection[random_selection_id]
        # 随机数处于个体对应的累计区间时，则将这个个体赋给新种群
        for i in range(population_size):
            if rand < pie_fitness[i]:
                new_population.append(population[i])
                break
        random_selection_id += 1

    return new_population


# 交叉函数，四个参数：种群个体、种群数量、交叉概率、维度信息
def crosscover(population, population_size, pc, dimen):
    for i in range(0, population_size - 1, 2):  # 每两个一对
        # 如果生成的随机数小于交叉概率
        if random.random() < pc:
            # 随机选择交叉点,random.randint(0, dimen-1),返回[0,dimen-1]之间的整数
            change_point = random.randint(0, dimen - 1)
            temp1 = []
            temp2 = []
            # 两个个体在交叉点进行交换
            temp1.extend(population[i][0: change_point])
            temp1.extend(population[i + 1][change_point:])
            temp2.extend(population[i + 1][0: change_point])
            temp2.extend(population[i][change_point:])
            population[i] = temp1
            population[i + 1] = temp2
    population = np.array(population)
    return population


# 变异函数，四个参数：种群个体、种群数量、变异概率、维度信息
def mutation(population, population_size, pm, dimen, start, end):
    for i in range(population_size):
        # 如果随机生成的数小于变异概率
        if random.random() < pm:
            # 随机生成个体中需要进行变异的数的数量mutation_num
            mutation_num = random.randint(0, dimen - 1)
            # 随机选择个体中mutation_num个数进行重新赋值
            for j in range(mutation_num):
                mutation_point = random.randint(0, dimen - 1)
                population[i][mutation_point] = np.random.uniform(low=start, high=end - 1)
    population = np.array(population)
    return population


def train(classify_mic, start, end, crossover, mutations):
    train_number = 0
    end += 1
    d = generate_load_balance_solution(classify_mic)
    cost = get_baseline(d, classify_mic)
    print("LB cost:", cost)

    d = generate_consolidation_solution(classify_mic)
    cost = get_baseline(d, classify_mic)
    print("Consolidation cost:", cost)

    while True:
        d = np.random.randint(start, end, size=len(classify_mic))
        if get_baseline(d, classify_mic) != 999999:
            baseline = get_baseline(d, classify_mic)
            break
    # 解的取值范围
    rangepop = [start, end]
    print("random cost: ", baseline)
    # return
    # 种群数量
    pn = 200
    # 迭代次数
    iterators = 1000
    # 交叉概率
    pc = crossover  # 0.9
    # 变异概率
    pm = mutations  # 0.2
    # 解的维度信息
    dimen = len(classify_mic)
    # 种群，为数组形式
    pop = np.zeros((pn, dimen), dtype=int)
    # 种群个体适应度值，为数组形式
    fitness = np.zeros(pn)
    # 随机初始化种群
    for j in range(pn):
        pop[j] = np.random.randint(low=start, high=end, size=dimen)
        # 计算适应度值
        fitness[j] = function(pop[j], classify_mic, baseline)
    # 获取当前最优解bestpop和最优适应度值bestfit
    bestpop, bestfit = pop[fitness.argmax()].copy(), fitness.max()
    # bestfitness保存每次迭代中的最优适应度值，为数组形式
    bestfitness = np.zeros(iterators)
    # 开始迭代训练

    d = np.random.randint(start, end, size=len(classify_mic))
    for i in tqdm(range(iterators)):
        # 选择操作
        parents = select(pop, pn, fitness)
        # 交叉操作
        crossover1 = crosscover(parents, pn, pc, dimen)
        # 变异操作
        pop = mutation(crossover1, pn, pm, dimen, start, end)
        # 将上一次迭代的最优适应度值和新的适应度值比较，选择更大的适应度值作为新的最优适应度值，对应的个体作为当前最优解
        for j in range(pn):
            # 确保个体取值在取值范围内[0,len(df_node['ID'])-1]内
            pop[pop < rangepop[0]] = rangepop[0]
            pop[pop > rangepop[1]] = rangepop[1]
            # 计算新个体适应度值
            fitness[j] = function(pop[j], classify_mic, baseline)
            if fitness[j] == 999999:
                pop[j] = adjust(pop[j], classify_mic, start, end)
                fitness[j] = function(pop[j], classify_mic, baseline)
            # 比较最优适应度值
            if fitness[j] < bestfit:
                train_number = i
                bestfit = fitness[j]
                bestpop = pop[j]
        bestfitness[i] = bestfit
        # if i % 100 == 0:
        #     print("当前迭代次数:", i)
        #     print("最优适应度值是:", bestfitness[i])
    print("迭代结束后:")
    print("最优解是:", bestpop)
    print("最优适应度值是:", bestfitness[-1])
    print("最优解具体值为：", get_baseline(bestpop, classify_mic))

    # d = np.random.randint(0, 4, size=12)
    number = 0
    new_baseline = 0
    while True:
        d = np.random.randint(start, end, size=len(classify_mic))
        if function(d, classify_mic, baseline) != 999999:
            new_baseline += function(d, classify_mic, baseline)
            number += 1
            # print(d)
            # print(function(d, classify_mic, baseline))
            if number == 10:
                print(new_baseline/10)
                print("效果提升: {}%".format((new_baseline - 10 * bestfitness[-1]) / new_baseline))
                return train_number
                # break

    # fig = plt.figure(figsize=(15, 10), dpi=50)
    # # plt.title('The Change of Best Fitness', fontdict={'weight': 'normal', 'size': 30})
    # x = range(1, 1001, 1)
    # plt.plot(x, bestfitness, color="red", linewidth=3.0, linestyle="-")
    # plt.tick_params(labelsize=25)
    # plt.xlabel("Epoch", fontdict={'weight': 'normal', 'size': 30})
    # plt.ylabel("Fitness value", fontdict={'weight': 'normal', 'size': 30})
    # plt.savefig("GA.pdf")
    # plt.show()
