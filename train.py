from data_helper import train_test_split, read_data, load_label
import os
import openai
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

openai.api_key = os.environ['OPEN_API_KEY']

# print(response)
# print(response['choices'][0]['message']['content'])

def train_model(train_content, train_label, test_content, save_path):
    length = len(train_content)
    random_int1 = np.random.randint(0, length - 1)
    random_int2 = np.random.randint(0, length - 1)
    for idx, content in enumerate(test_content):
        input_msg = [
            {"role": "system", "content": "你现在是一个辅助我进行自然语言处理训练的AI助手。接下来的回答中，你只需要回答1，-1，或者0，分别代表情感语义中的积极，消极，和中性"}
        ]
        input_msg += [
            {"role": "user", "content": train_content[random_int1]},
            {"role": "assistant", "content": str(train_label[random_int1])},
            {"role": "user", "content": train_content[random_int2]},
            {"role": "assistant", "content": str(train_label[random_int2])}
        ]
        input_msg.append(
            {"role": "user", "content": content}
        )
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = input_msg
        )
        print(response)
        with open(save_path, 'a+', encoding='utf-8-sig') as f:
            f.write(response['choices'][0]['message']['content']+',')
        time.sleep(3)


def eval_data(real_list, predict_list):
    real_list = [int(y) for y in real_list]
    predict_list = [int(x) for x in predict_list]
    accuracy = accuracy_score(real_list, predict_list)
    precision_weighted = precision_score(real_list, predict_list, average='weighted')
    recall_weighted = recall_score(real_list, predict_list, average='weighted')
    f1_score_weighted = f1_score(real_list, predict_list, average='weighted')
    print('准确率：', accuracy)
    print('精确率：', precision_weighted)
    print('召回率：', recall_weighted)
    print('F1-score：', f1_score_weighted)
    report = classification_report(y_true=real_list, y_pred=predict_list, labels=[-1, 0, 1])
    print(report)

if __name__ == '__main__':
    p_path = './temp/preidict.txt'
    r_path = './temp/real.txt'
    if os.path.exists(p_path) and os.path.exists(r_path):
        real, predict = load_label()
        eval_data(real, predict)
    else:
        train_test_split()
        train_content, train_label, test_content, test_label = read_data()
        with open(r_path, 'w', encoding='utf-8-sig') as r:
            r.write(','.join([str(label) for label in test_label]))
        train_model(train_content, train_label, test_content, p_path)
    