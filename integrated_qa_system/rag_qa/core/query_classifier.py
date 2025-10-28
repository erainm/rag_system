# Created by erainm on 2025/10/27 13:23.
# IDE：PyCharm 
# @Project: rag_system
# @File：query_classifier
# @Description: 查询意图识别

import json
import os
import torch
import numpy as np
import sys
from base.logger import logger
from base.config import conf
# 导入Transformer库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(os.path.abspath(current_dir)))
project_root = os.path.abspath(os.path.dirname(os.path.abspath(rag_qa_path)))
sys.path.insert(0, project_root)

"""
意图识别模块：
    1. 数据加载：读取JSON数据集，包含查询和标签（“通用知识”或“专业知识”）
    2. 模型训练：使用bert-base-chinese模型，微调二分类任务，准确率达90%+
    3. 评估优化：直接处理数字标签（0、1），生成分类报告和混淆矩阵
    4. 预测接口：支持实时分类，集成到EduRAG系统
    
实现：
    1. 初始化方法：初始化预训练的分词器、预训练模型。如果是上线阶段，主要负责加载训练好的模型
    2. 数据预处理：将查询文本和预训练标签转化为模型需要的输入数据格式
    3. 构建数据集：用于模型训练，适配模型的训练函数
    4. 模型训练：基于处理好的数据集划分出来训练集，对模型进行训练
    5. 模型评估：在数据集划分出来的验证集，对模型进行评估
    6. 模型预测：加载训练好的模型，完成意图识别任务
"""

class QueryClassifier(object):
    """
    一、初始化方法：初始化预训练的分词器、预训练模型。如果是上线阶段，主要负责加载训练好的模型
        1. 获取bert预训练模型所在目录
        2. 加载预训练模型分析器
        3. 设置训练设备
        4. 定义标签映射
        5. 尝试加载模型
    """
    def __init__(self, model_path='/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/models/bert_query_classifier'):
        print(f"rag_qa_path ---> {rag_qa_path}")
        self.pre_trained_model_path = '/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/models/bert-base-chinese'
        self.model_path = f'{rag_qa_path}/{model_path}'
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_path, local_files_only=True)

        self.model = None
        self.device = "cpu"

        logger.info(f"使用设备：{self.device},开始加载模型……")
        # 定义标签映射
        self.label_map = {"通用知识": 0,  "专业咨询": 1}
        # 尝试加载模型
        self.load_model()

    def load_model(self):
        # 检查模型路径是否存在
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 初始化新模型
            self.model = BertForSequenceClassification.from_pretrained("/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/models/bert-base-chinese", num_labels=2)
            # print(f'self.model--》{self.model}')
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")
    """
    二、数据预处理：将查询文本和预训练标签转化为模型需要的输入数据格式
        1. 接收传入的query数据和分类标签数据
        2. 将查询文本数值化，使用预训练的tokenizer对query进行编码和长度补齐，得到input_ids和attention_mask
        3. 将标签数据数值化
    """
    def preprocess_data(self, query, labels):
        encodings = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        return encodings, [self.label_map[label] for label in labels]

    """
    三、构建数据集：用于模型训练，适配模型的训练函数
        1. 预处理的数据化query和label数据
        2. 集成实现Dataset
            2.1 实现初始化方法
            2.2 重写__getitem__方法，返回指定索引的数据
            2.3 重写__len__方法，返回数据集的长度
        3. 构建Dataset类并返回
    """
    def build_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        return Dataset(encodings, labels)

    """
    四、模型训练：基于处理好的数据集划分出来训练集，对模型进行训练
        1. 划分数据集
        2. 构建数据集
        3. 构建训练参数
        4. 构建Trainer
        5. 训练模型
    """
    def train_model(self, data_file="training_dataset_hybrid_5000.json"):
        """训练 BERT 分类模型"""
        # 加载数据集
        # print(f'os.path.exists(data_file)---》{os.path.exists(data_file)}')
        if not os.path.exists(data_file):
            logger.error(f"数据集文件 {data_file} 不存在")
            raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

        with open(data_file, "r", encoding="utf-8") as f:
            # print(f'f.readlines()--》{f.readlines()}')
            data = [json.loads(value) for value in f.readlines()]
            # print(f'data--》{data}')
            # print(f'data--》{type(data[0])}')

        texts = [item["query"] for item in data]
        # print(f'texts--》{texts[:2]}')
        # print(f'样本个数texts--》{len(texts)}')
        labels = [item["label"] for item in data]
        # print(f'labels--》{labels[:2]}')
        # 数据划分
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        # print(f'train_texts--》{len(train_texts)}')
        # print(f'val_texts--》{len(val_texts)}')
        # 数据预处理
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        # 得到dataset对象
        train_dataset = self.build_dataset(train_encodings, train_labels)
        val_dataset = self.build_dataset(val_encodings, val_labels)
        # print('取出一个样本', train_dataset[1])
        # print(f'验证集样本的个数--》{len(train_dataset)}')
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_result",  # 模型（检查点）以及日志保存的路径等，
            num_train_epochs=3,  # 训练的轮次
            per_device_train_batch_size=8,  # 训练的批次
            per_device_eval_batch_size=8,  # 验证批次
            warmup_steps=20,  # 学习率预热的步数
            weight_decay=0.01,  # 权重衰减系数
            logging_dir="./bert_logs",  # 日志保存路径:如果想生成这个文件夹，需要安装tensorboard
            logging_steps=10,  # 每隔多少步打印日志
            eval_strategy="epoch",  # 每轮都进行评估
            save_strategy="epoch",  # 每轮都进行检查点的模型保存
            load_best_model_at_end=True,  # 加载最优的模型
            save_total_limit=1,  # 只保存一个检查点，其他被覆盖
            metric_for_best_model="eval_loss",  # 评估最优模型的指标（验证集损失）
            fp16=False,  # 禁用混合精度
        )

        # print(f'training_args--》{training_args}')
        # 初始化Trainer
        trainer = Trainer(model=self.model, args=training_args,
                          train_dataset=train_dataset, eval_dataset=val_dataset,
                          compute_metrics=self.compute_metrics)
        # 开始训练模型
        logger.info("开始训练BERT模型")
        trainer.train()
        self.save_model()

        # 对验证集进行训练好的模型验证
        # val_texts-->原始的文本；val_labels--是标签数字
        self.evaluate_model(val_texts, val_labels)


    def compute_metrics(self, eval_pred):
        """
        计算评估指标
            logits：预测权重值
            labels：真实值
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    """
    五、模型评估：在数据集划分出来的验证集，对模型进行评估
        1. 数据预处理
            1.1 对输入文本进行分词编码（截断/填充至128长度）
            1.2 创建包含编码和标签的Torch数据集
        2. 初始化预测工具
            2.1 创建Trainer实例加载当前模型
        3. 执行预测
            3.1 使用predict方法获取原始预测结果
            3.2 通过argmax解析预测标签得到概率最大的预测值的索引
        4. 生成评估报告
            4.1 输出分类报告
            4.2 输出混淆矩阵
    """
    def evaluate_model(self, texts, labels):
        """评估模型性能"""
        # 仅对 texts 进行分词，labels 已为数字
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        dataset = self.build_dataset(encodings, labels)
        print(f'len(dataset)-->{len(dataset)}')
        print(f'dataset[0]--->{dataset[0]}')
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(dataset)
        print(f'predictions--》{predictions}')
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        # print(f'pred_labels--》{type(pred_labels)}')
        # print(f'predictions.label_ids--》{predictions.label_ids}')
        true_labels = labels

        logger.info("分类报告:")
        logger.info(classification_report(
            true_labels,
            pred_labels,
            target_names=["通用知识", "专业咨询"]
        ))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(true_labels, pred_labels))

    """
    六、模型预测：基于训练好的模型，对用户输入的查询进行预测
        1. 构建模型
        2. 加载模型
        3. 构建输入数据
        4. 预测
    """
    def predict_category(self, query):
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # print(f'outputs--》{outputs}')
            # logits = outputs.logits
            # print(f'logits--》{logits}')
            # # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"

    def save_model(self):
        """保存模型"""
        self.model.save_pretrained("/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/models/bert_query_classifier")
        self.tokenizer.save_pretrained("/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/models/bert_query_classifier")
        logger.info(f"模型保存至: ../models/bert_query_classifier")



if __name__ == '__main__':
    query_classifier_model = QueryClassifier(model_path='models/bert_query_classifier')
    path = f'/Users/erainm/Documents/application/dev/workSpace/rag_system/integrated_qa_system/rag_qa/data/classify_data/model_generic_5000.json'
    # with open(path, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     query = []
    #     labels = []
    #     for line in lines:
    #         data = json.loads(line)
    #         query.append(data['query'])
    #         labels.append(data['label'])
    # print(len(query), len(labels))
    #
    # encodings, labels = query_classifier_model.preprocess_data(query, labels)
    # print(encodings, labels)
    # datasets = query_classifier_model.build_dataset(encodings, labels)
    # 模型训练
    # query_classifier_model.train_model(path)

    # 模型预测
    test_queries = [
        "如何查询天气",
        "AI学科的课程大纲是什么？",
        "Java怎么学"
     ]

    for query in test_queries:
        category = query_classifier_model.predict_category(query)
        print(f"查询：{query}，类别：{category}")
