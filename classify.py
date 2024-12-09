from importlib import import_module
import numpy as np
import torch
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import DPCNN
import argparse
from utils import build_dataset, build_iterator

parser = argparse.ArgumentParser(description='文本分类')
parser.add_argument('--input', type=str, required=True, help='需要分类的文件路径')
parser.add_argument('--output', type=str, required=True, help='保存分类结果的文件路径')
args = parser.parse_args()

if __name__ == '__main__':
    
    # 加载DPCNN分类模型
    start_time = time.time()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    config = DPCNN.Config()
    model = DPCNN.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    
    # 加载测试数据
    config.test_path = args.input
    test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)
    
    # 测试包括未知类别的测试集
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts in test_iter:
            outputs = model(texts)          
            probs = torch.softmax(outputs, dim=1)  # 输出类别的概率
            max_probs, predic = torch.max(probs, 1)  # 获取最大概率及对应的类别
            predict_all = np.append(predict_all, predic)
    
    # 使用未知分类判断器判断描述是否属于已知分类
    with open('./data/known_class_means.pkl','rb') as file:
        known_class_means = pickle.load(file)
    model = SentenceTransformer('./data/text2vec-base-chinese')
    threshold = 0.65  # 余弦相似度的阈值
    
    def detect_unknown_class(embedding):
        max_sim = max(
            np.dot(embedding, known_mean) / (np.linalg.norm(embedding) * np.linalg.norm(known_mean))
            for known_mean in known_class_means.values()
        )
        return max_sim < threshold
    
    y_pred = []
    df = pd.read_excel(args.input)
    X_test = df["描述"].tolist()
    
    # 记录文件中描述为空的行索引
    nan_rows = df[df.isnull().any(axis=1)].index.tolist()
    
    X_test_str = [str(x) for x in X_test]
    X_test_embeddings = model.encode(X_test_str)
    for embedding in X_test_embeddings:
        if(detect_unknown_class(embedding)):
            y_pred.append("未知分类")
        else:
            y_pred.append("")
            
    # 合并未知分类判断器和已知分类识别器的分类结果
    with open('./data/class.txt',encoding='utf-8') as file:
        label_list = []
        for line in file.readlines():
            label = line.strip()
            label_list.append(label)
            
    y_pred_end = []
    for y1, y2 in zip(y_pred,predict_all):
        if y1=='未知分类':
            y_pred_end.append(y1)
        else:
            y_pred_end.append(label_list[y2])
    for i in nan_rows:
        y_pred_end[i] = None
    df['预测分类'] = y_pred_end
    df.to_excel(args.output, index=False)
    print("分类完成")