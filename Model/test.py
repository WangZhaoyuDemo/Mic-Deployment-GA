import data_process
import util
import GA
import numpy as np

if __name__ == '__main__':
    # 读取数据集
    df_node, df_mic, df_layer, df_callGraph = data_process.build_data_df(3)
    mic, layer, node, call_graph = data_process.data_process(df_node, df_mic, df_layer, df_callGraph)

    # 建立通信网络矩阵
    bandwidth_matrix = util.build_bandwidth_matrix(node)

    # 建立调用图邻接矩阵和所有连通分量
    communicate_matrix, node_mapping = util.build_adjacency_matrix(call_graph)
    split_mic_list = util.split_graph(communicate_matrix)
    result = util.build_classify_mic(split_mic_list, node_mapping, mic, node, df_mic)

    # GA.train(result[2]['classiyf_mic'], result[2]['start'], result[2]['end'])
    Crosscover = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    Mutation = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    i = result[0]
    Crosscover_result = []
    Mutation_result = []
    for j in Crosscover:
        Crosscover_result.append(GA.train(i['classiyf_mic'], i['start'], i['end'], j, 0.2))
    for j in Crosscover_result:
        print(j)
    print("---------------------------------")
    for j in Mutation:
        Mutation_result.append(GA.train(i['classiyf_mic'], i['start'], i['end'], 0.9, j))
    for j in Crosscover_result:
        print(j)
    print("---------------------------------")
    for j in Mutation_result:
        print(j)
