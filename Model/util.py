import numpy as np


def build_bandwidth_matrix(node):
    """
    :param node: 节点列表
    :return: 通信矩阵
    """
    bandwidth_matrix = np.random.randint(50, 100, size=(len(node), len(node)))
    bandwidth_matrix = ((bandwidth_matrix + bandwidth_matrix.T) / 2).astype(int)
    # 构建网络通信图和调用关系图
    for i in range(len(node)):
        bandwidth_matrix[i][i] = 9999
    return bandwidth_matrix


def build_adjacency_matrix(call_graph):
    """
    :param call_graph: 微服务调用图
    :return: 微服务的邻接矩阵和节点映射关系
    """
    edges = []
    for i in call_graph:
        edges.append((i['Microservice1 ID'], i['Microservice2 ID'], i['Data Size']))
    node_mapping = {}
    # 使用字典动态记录节点的映射关系
    num_nodes = 0

    # 构建节点映射关系，并记录节点数量
    for edge in edges:
        start_node, end_node, data_size = edge
        if start_node not in node_mapping:
            node_mapping[start_node] = num_nodes
            num_nodes += 1
        if end_node not in node_mapping:
            node_mapping[end_node] = num_nodes
            num_nodes += 1

    # 初始化邻接矩阵
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # 添加边到邻接矩阵
    for edge in edges:
        start_node, end_node, data_size = edge
        start_idx = node_mapping[start_node]
        end_idx = node_mapping[end_node]
        adjacency_matrix[start_idx][end_idx] = data_size

    return adjacency_matrix, node_mapping


def split_graph(adj_matrix):
    """
    :param adj_matrix: 邻接矩阵
    :return: 邻接矩阵的所有连通分量
    """
    def dfs(node, current_subgraph):
        visited[node] = True
        current_subgraph.append(node)

        for neighbor in range(len(adj_matrix)):
            if (adj_matrix[node][neighbor] != 0 or adj_matrix[neighbor][node] != 0) and not visited[neighbor]:
                dfs(neighbor, current_subgraph)

    n = len(adj_matrix)
    visited = [False] * n
    subgraphs = []

    for node in range(n):
        if not visited[node]:
            current_subgraph = []
            dfs(node, current_subgraph)
            subgraphs.append(current_subgraph)

    return subgraphs


def build_classify_mic(split_mic_list, node_mapping, mic, node, df_mic):
    start = 0
    end = -1
    mic_number = 0
    # main()
    node_mapping_hashmap = {}
    for key, value in node_mapping.items():
        node_mapping_hashmap[value] = key
    for mic_list in split_mic_list:
        mic_number += len(mic_list)
    result = []
    for mic_list in split_mic_list:
        classify_mic = {}
        i = 0
        start = end + 1
        end = int(start + len(node) * len(mic_list) / mic_number)
        if end >= len(node):
            end = len(node) - 1
        for m in mic:
            if node_mapping[m['Microservice ID']] in mic_list:
                classify_mic[i] = m['ID']
                i += 1
        print(start, end, len(node))
        print(mic_list)
        # for i in classify_mic:
        #     print(classify_mic[i], mic[classify_mic[i]]['Microservice ID'])
        # print("      ")
        # for m in mic_list:
        #     print(m, node_mapping_hashmap[m])
        # print("      ")
        temp_dic = {'classiyf_mic': classify_mic, 'start': start, 'end': end}
        result.append(temp_dic)
    return result
