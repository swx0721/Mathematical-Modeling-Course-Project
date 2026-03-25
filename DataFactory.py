import networkx as nx
import pandas as pd
import numpy as np

def generate_campus_data(n_students=2000, avg_contacts=10):
    """
    生成模拟校园社交网络
    n_students: 学生人数
    avg_contacts: 平均每个学生每天密切接触的人数
    """
    # 使用Watts-Strogatz小世界模型模拟校园：既有宿舍/班级小圈子，也有跨学科社交
    # p=0.1 代表有10%的概率产生跨圈子接触
    G = nx.watts_strogatz_graph(n=n_students, k=avg_contacts, p=0.1)
    
    # 导出边列表 (Edge List)
    edge_list = nx.to_pandas_edgelist(G)
    edge_list.to_csv("campus_network.csv", index=False)
    
    # 为学生分配随机的疫苗接种状态 (假设20%接种率)
    vax_status = np.random.choice([0, 1], size=n_students, p=[0.8, 0.2])
    pd.DataFrame({'student_id': range(n_students), 'is_vaccinated': vax_status}).to_csv("student_metadata.csv", index=False)
    
    print("成功生成校园模拟数据集: campus_network.csv & student_metadata.csv")

if __name__ == "__main__":
    generate_campus_data()