import data_process
import util
import GA
import numpy as np
if __name__ == '__main__':
    # 读取数据集
    df_node, df_mic, df_layer, df_callGraph = data_process.build_data_df(2)
    mic, layer, node, call_graph = data_process.data_process(df_node, df_mic, df_layer, df_callGraph)

    # 建立通信网络矩阵
    bandwidth_matrix = util.build_bandwidth_matrix(node)

    # 建立调用图邻接矩阵和所有连通分量
    communicate_matrix, node_mapping = util.build_adjacency_matrix(call_graph)
    split_mic_list = util.split_graph(communicate_matrix)
    result = util.build_classify_mic(split_mic_list, node_mapping, mic, node, df_mic)

    # for i in result:
    #     GA.train(i['classiyf_mic'], i['start'], i['end'], 0.9, 0.2)

    # GA.train(result[2]['classiyf_mic'], result[2]['start'], result[2]['end'])
    # crossover = 0.0
    # while True:
    #     crossover += 0.1
    #     if crossover >= 1.0:
    #         break
    #     print("------------------------------------------------")
    #     print("crossover = ", crossover)
    #     for i in result:
    #         GA.train(i['classiyf_mic'], i['start'], i['end'], crossover, 0.2)

    mutations = 0.0
    while True:
        mutations += 0.1
        if mutations >= 1.0:
            break
        print("------------------------------------------------")
        print("crossover = ", mutations)
        for i in result:
            GA.train(i['classiyf_mic'], i['start'], i['end'], 0.9, mutations)
      