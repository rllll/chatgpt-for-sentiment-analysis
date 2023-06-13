import pandas as pd
import numpy as np
import os

def train_test_split(split_scale = 0.3):
    train_data_file, test_data_file = './data/train.csv', './data/test.csv'
    if os.path.exists(train_data_file) and os.path.exists(test_data_file):
        return
    # type_list = ['宝马1系', '宝马2系', '宝马3系（1）','宝马3系（2）', '宝马3系（3）', '宝马3系（4）','宝马3系（5）', '宝马3系（6）', '宝马3系（7）','宝马3系（8）', '宝马3系（9）', '宝马3系（10)','宝马3系（11）', '宝马3系（12）', '宝马4系','宝马5系','宝马X1（1）', '宝马X1（2）', '宝马X1(5)', '宝马X2', '宝马X3']
    type_list = ['宝马2系']
    df = pd.read_excel('./data/bmw_all.xlsx', sheet_name=type_list, usecols=[2, 21], dtype=str)
    contents, labels = [], []
    for type in type_list:
        df_drop = df[type].dropna(subset=['总的来说'])
        contents += df_drop["具体评价"].tolist()
        labels += df_drop['总的来说'].tolist()
    cup_list = []
    for idx, label in enumerate(labels):
        cup_list.append({
            'content': contents[idx],
            'label': label
        })
    # 利用洗牌算法打乱
    np.random.shuffle(cup_list)
    divpos = round((1 - split_scale) * len(cup_list))
    train_cup, test_cup = cup_list[:divpos], cup_list[divpos:]
    train_text, train_label = [], []
    for item in train_cup:
        train_text.append(item['content'])
        train_label.append(int(item['label']))
    test_text, test_label = [], []
    for item in test_cup:
        test_text.append(item['content'])
        test_label.append(int(item['label']))
    
    train_df = pd.DataFrame({
        'content': train_text,
        'label': train_label
    })
    test_df = pd.DataFrame({
        'content': test_text,
        'label': test_label
    })

    train_df.to_csv(train_data_file, encoding='utf-8-sig', index=None)
    test_df.to_csv(test_data_file, encoding='utf-8-sig', index=None)

def read_data():
    train_data_file, test_data_file = './data/train.csv', './data/test.csv'
    train_df = pd.read_csv(train_data_file, encoding='utf-8-sig')
    test_df = pd.read_csv(test_data_file, encoding='utf-8-sig')
    train_content, train_label = train_df['content'].tolist(), train_df['label'].tolist()
    test_content, test_label = test_df['content'].tolist(), test_df['label'].tolist()
    return train_content, train_label, test_content, test_label

def load_label():
    predict_path = './temp/preidict.txt'
    real_path = './temp/real.txt'
    p_list, r_list = [], []
    with open(predict_path, 'r', encoding='utf-8-sig') as p:
        p_list = str(p.read()).split(',')
    with open(real_path, 'r', encoding='utf-8-sig') as r:
        r_list = str(r.read()).split(',')
    predict_list = []
    for cur in p_list:
        if cur.startswith('-1'):
            predict_list.append('-1')
        elif cur.startswith('0'):
            predict_list.append('0')
        elif cur.startswith('1'):
            predict_list.append('1')
        elif cur == '':
            continue
        else:
            predict_list.append('2')
    return r_list, predict_list