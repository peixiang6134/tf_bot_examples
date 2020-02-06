import os
import json
import random
from collections import Counter


def process(path, out_dir, dev_size=1000, test_size=1000, seed=2020):
    """
    读入数据，并划分训练集，开发集，测试集

    :param path: 原始数据的路径
    :param out_dir: 用于写入结果的目录
    :param dev_size: 指定验证集的大小
    :param test_size: 指定测试集的大小
    :param seed: 随机数种子
    """

    with open(path) as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)  # 随机打乱数据

    write('test', data[:test_size], out_dir)
    write('dev', data[test_size:test_size+dev_size], out_dir)
    write('train', data[test_size+dev_size:], out_dir)


def write(split, data, out_dir):
    """将某个split的数据写入到out_dir

    :param split: str, train,dev,test 三者之一
    :param data: list, 需要写入的数据
    :param out_dir: str, 用于写入结果的目录
    """
    src = open(os.path.join(out_dir, f'{split}.src'), 'w')
    tgt = open(os.path.join(out_dir, f'{split}.tgt'), 'w')
    print(f"writing {split} data to {out_dir}...")
    for text, label in data:
        text = text.replace("\t", "").replace("\n", "")  # 去掉\t \n字符
        src.write(f"{text}\n")
        tgt.write(f"{label}\n")
    src.close()
    tgt.close()


def build_vocab(data_path, max_size=20000):
    """根据data_path中的数据构建字典, max_size是字典的最大容量"""
    vocab = ["<blank>", "<s>", "</s>", "<unk>"]
    counter = Counter()
    with open(data_path) as f:
        for line in f:
            words = [word for word in line.strip().split() if word]
            counter.update(words)

    for word, _ in counter.most_common():
        vocab.append(word)
        if len(vocab) >= max_size:
            break

    print("writing vocab to voc.txt....")
    with open('voc.txt', 'w') as f:
        f.write('\n'.join(vocab))


def main():
    path = "./data/train.json"
    out_dir = "./data/processed"
    os.makedirs(out_dir, exist_ok=True)
    process(path, out_dir)
    build_vocab("./data/processed/train.src")


if __name__ == "__main__":
    main()
