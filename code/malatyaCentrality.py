import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Users/efe44/Desktop/veri madenciliği proje/Clustered_OnlineRetail.xlsx")

df['InvoiceYear'] = df['InvoiceDate'].dt.year
df['InvoiceMonth'] = df['InvoiceDate'].dt.month.astype(str).str.zfill(2)
df['InvoiceDay'] = df['InvoiceDate'].dt.day
df['InvoiceTime'] = df['InvoiceDate'].dt.strftime('%H:%M:%S')

features = df[['UnitPrice', 'Quantity', 'InvoiceYear', 'InvoiceMonth', 'InvoiceDay', 'InvoiceTime', 'Country', 'Cluster']]

rules = {
    "InvoiceMonth": {
        "1. çeyrek": ["01", "02", "03"],
        "2. çeyrek": ["04", "05", "06"],
        "3. çeyrek": ["07", "08", "09"],
        "4. çeyrek": ["10", "11", "12"]
    },
    "Quantity": {
        "10 ve daha fazla": lambda x: x >= 10,
        "10'dan az": lambda x: 0 <= x < 10,
        "aykırı": lambda x: x < 0
    },
    "UnitPrice": {
        "ucuz": lambda x: 0 <= x < 5,
        "orta": lambda x: 5 <= x < 10,
        "pahalı": lambda x: x >= 10
    },
    "InvoiceTime": {
        "sabah": lambda x: 6 <= int(x.split(':')[0]) < 11,
        "öğlen": lambda x: 11 <= int(x.split(':')[0]) < 15,
        "akşam": lambda x: int(x.split(':')[0]) >= 15
    }
}

def categorize(value, rule):
    for label, condition in rule.items():
        if callable(condition):
            if condition(value):
                return label
        elif value in condition:
            return label
    return None

def calculate_malatya_centrality(graph, node):
    degree_node = sum(data['weight'] for _, _, data in graph.edges(node, data=True))
    neighbor_degrees = sum(
        sum(data['weight'] for _, _, data in graph.edges(neighbor, data=True))
        for neighbor in graph.neighbors(node)
    )
    return degree_node / neighbor_degrees if neighbor_degrees > 0 else 0

new_sample = {
    "UnitPrice": 16.63,
    "Quantity": 1,
    "InvoiceYear": 2011,
    "InvoiceMonth": "05",
    "InvoiceDay": 10,
    "InvoiceTime": "15:09:00",
    "Country": "United Kingdom"
}

clusters = features['Cluster'].unique()
cluster_graphs = {}

for cluster in clusters:
    
    cluster_data = features[features['Cluster'] == cluster].sample(
        min(10, len(features[features['Cluster'] == cluster])),
        random_state=1
    )  

    G = nx.Graph()

    for idx, row in cluster_data.iterrows():
        G.add_node(idx, **row.to_dict())

    for idx1, row1 in cluster_data.iterrows():
        for idx2, row2 in cluster_data.iterrows():
            if idx1 >= idx2:
                continue

            common_features = 0

            if categorize(row1['UnitPrice'], rules['UnitPrice']) == categorize(row2['UnitPrice'], rules['UnitPrice']):
                common_features += 1

            if categorize(row1['Quantity'], rules['Quantity']) == categorize(row2['Quantity'], rules['Quantity']):
                common_features += 1

            if categorize(row1['InvoiceMonth'], rules['InvoiceMonth']) == categorize(row2['InvoiceMonth'], rules['InvoiceMonth']):
                common_features += 1

            if categorize(row1['InvoiceTime'], rules['InvoiceTime']) == categorize(row2['InvoiceTime'], rules['InvoiceTime']):
                common_features += 1

            if row1['InvoiceYear'] == row2['InvoiceYear']:
                common_features += 1

            if row1['InvoiceDay'] == row2['InvoiceDay']:
                common_features += 1

            if row1['Country'] == row2['Country']:
                common_features += 1

            if common_features > 0:
                G.add_edge(idx1, idx2, weight=common_features)

    new_node_id = len(G.nodes)
    G.add_node(new_node_id, **new_sample)

    for idx, row in cluster_data.iterrows():
        common_features = 0

        if categorize(new_sample['UnitPrice'], rules['UnitPrice']) == categorize(row['UnitPrice'], rules['UnitPrice']):
            common_features += 1

        if categorize(new_sample['Quantity'], rules['Quantity']) == categorize(row['Quantity'], rules['Quantity']):
            common_features += 1

        if categorize(new_sample['InvoiceMonth'], rules['InvoiceMonth']) == categorize(row['InvoiceMonth'], rules['InvoiceMonth']):
            common_features += 1

        if categorize(new_sample['InvoiceTime'], rules['InvoiceTime']) == categorize(row['InvoiceTime'], rules['InvoiceTime']):
            common_features += 1

        if new_sample['InvoiceYear'] == row['InvoiceYear']:
            common_features += 1

        if new_sample['InvoiceDay'] == row['InvoiceDay']:
            common_features += 1

        if new_sample['Country'] == row['Country']:
            common_features += 1

        if common_features > 0:
            G.add_edge(new_node_id, idx, weight=common_features)

    malatya_centrality = calculate_malatya_centrality(G, new_node_id)

    cluster_graphs[cluster] = G

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    node_colors = ['skyblue' if node != new_node_id else 'orange' for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)

    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in edges])

    edge_labels = {(u, v): d['weight'] for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(f'Cluster {cluster} Graph\nMalatya Centrality: {malatya_centrality:.4f}')
    plt.show()
