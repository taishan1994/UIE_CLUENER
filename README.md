# UIE_CLUENER
用百度的UIE解决CLUENER2020细粒度实体识别数据集。

# 安装依赖

```python
python -m pip install paddlepaddle-gpu==2.4.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddlenlp
pip install faster_tokenizer
#paddlepaddle==2.4.1
#paddlenlp==2.4.5
```

# 数据准备

以下数据处理都已完成，这里作说明，可以跳过到训练那块。

原始数据为：data/cluener/下的train.json、dev.json和test.json，其中trian.json和dev.json里面的数据格式为：

```python
{"text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，", "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}}
```

test.json里面只有数据，没有对应的标签，格式为：

```python
{"id": 0, "text": "四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。"}
```

进入到text目录下。process.py里面有两个函数，process_cluener_main()用于将数据处理成doccano标注的格式，show_cluener_doccano_txt()后面再讲，可以在运行时先注释掉。运行后得到data/cluener下的train_doccano.json和dev_doccano.json，里面的数据格式为：

```python
{"id": 0, "text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，", "relations": [], "entities": [{"id": 0, "label": "人名", "start_offset": 0, "end_offset": 3}, {"id": 1, "label": "地址", "start_offset": 15, "end_offset": 17}]}
```

并且得到数据的相关信息：不同实体的数目以及句子长度对应的数目

```python
+--------+------+
| 标签名 | 数目 |
+--------+------+
|  组织  | 3075 |
|  人名  | 3661 |
|  地址  | 2829 |
|  公司  | 2897 |
|  政府  | 1797 |
|  书籍  | 1131 |
|  游戏  | 2325 |
|  电影  | 1109 |
|  职位  | 3052 |
|  景点  | 1462 |
+--------+------+
+----------+------+
| 文本长度 | 数目 |
+----------+------+
|    2     |  1   |
|    3     |  7   |
|    4     |  8   |
|    5     |  12  |
|    6     |  16  |
|    7     |  10  |
|    8     |  26  |
|    9     |  37  |
|    10    |  51  |
|    11    |  80  |
|    12    |  69  |
|    13    |  62  |
|    14    |  77  |
|    15    |  89  |
|    16    | 114  |
|    17    | 159  |
|    18    | 121  |
|    19    | 112  |
|    20    | 141  |
|    21    | 134  |
|    22    | 208  |
|    23    |  91  |
|    24    |  82  |
|    25    | 124  |
|    26    |  95  |
|    27    |  97  |
|    28    | 133  |
|    29    | 117  |
|    30    | 165  |
|    31    | 154  |
|    32    | 193  |
|    33    | 200  |
|    34    | 213  |
|    35    | 259  |
|    36    | 290  |
|    37    | 338  |
|    38    | 391  |
|    39    | 395  |
|    40    | 471  |
|    41    | 487  |
|    42    | 537  |
|    43    | 564  |
|    44    | 538  |
|    45    | 581  |
|    46    | 504  |
|    47    | 516  |
|    48    | 571  |
|    49    | 595  |
|    50    | 513  |
+----------+------+
====================================================================================================
+--------+------+
| 标签名 | 数目 |
+--------+------+
|  组织  | 344  |
|  人名  | 451  |
|  地址  | 364  |
|  公司  | 366  |
|  政府  | 244  |
|  书籍  | 152  |
|  游戏  | 287  |
|  电影  | 150  |
|  职位  | 425  |
|  景点  | 199  |
+--------+------+
+----------+------+
| 文本长度 | 数目 |
+----------+------+
|    2     |  2   |
|    3     |  2   |
|    4     |  1   |
|    6     |  4   |
|    7     |  5   |
|    8     |  4   |
|    9     |  6   |
|    10    |  5   |
|    11    |  12  |
|    12    |  8   |
|    13    |  6   |
|    14    |  7   |
|    15    |  7   |
|    16    |  7   |
|    17    |  12  |
|    18    |  17  |
|    19    |  12  |
|    20    |  15  |
|    21    |  17  |
|    22    |  31  |
|    23    |  14  |
|    24    |  18  |
|    25    |  9   |
|    26    |  6   |
|    27    |  16  |
|    28    |  14  |
|    29    |  16  |
|    30    |  16  |
|    31    |  23  |
|    32    |  26  |
|    33    |  26  |
|    34    |  35  |
|    35    |  44  |
|    36    |  29  |
|    37    |  45  |
|    38    |  44  |
|    39    |  60  |
|    40    |  59  |
|    41    |  57  |
|    42    |  63  |
|    43    |  72  |
|    44    |  55  |
|    45    |  85  |
|    46    |  57  |
|    47    |  62  |
|    48    |  73  |
|    49    |  78  |
|    50    |  61  |
+----------+------+
====================================================================================================
```

接下来运行：

```python
python doccano.py \
    --doccano_file ../data/cluener/train_doccano.json \
    --task_type ext \
    --save_dir ../data/cluener/ \
    --splits 0.9 0.0 0.1

mv ../data/cluener/train.txt ../data/cluener/train1.txt
mv ../data/cluener/test.txt ../data/cluener/test1.txt

python doccano.py \
    --doccano_file ../data/cluener/dev_doccano.json \
    --task_type ext \
    --save_dir ../data/cluener/ \
    --splits 0.0 1.0 0.0

rm -f ../data/cluener/train.txt
rm -f ../data/cluener/test.txt
mv ../data/cluener/train1.txt ../data/cluener/train.txt
mv ../data/cluener/test1.txt ../data/cluener/test.txt
```

因为doccano.py一定会生成train.txt、dev.txt和test.txt，对dev_doccano.json处理时为了避免覆盖，采取了先重命名再改回来的策略。如果是windows系统，可以自己手动对文件进行修改。

train.txt、dev.txt和test.txt是训练需要的数据，里面的格式为：

```python
{"content": "E32009将于下月第一周在洛杉矶会展中心举行，并正式向普通民众开放参观，", "result_list": [{"text": "E32009", "start": 0, "end": 6}], "prompt": "组织"}
```

至此，数据就处理好了。

# 训练和验证模型

训练模型：

```python
python finetune.py  \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir ./checkpoint/model_best \
    --train_path ../data/cluener/train.txt \
    --dev_path ../data/cluener/dev.txt  \
    --max_seq_len 64  \
    --per_device_train_batch_size  128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir ./checkpoint/model_best \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```

验证模型：

```python
python evaluate.py \
    --model_path ./checkpoint/model_best/checkpoint-1800/ \
    --test_path ../data/cluener/dev.txt \
    --batch_size 128 \
    --max_seq_len 64

"""
[2022-12-22 06:36:12,652] [    INFO] - Evaluation Precision: 0.81694 | Recall: 0.81641 | F1: 0.81667
"""
```

展示每类实体的分数：

```python
python evaluate.py \
    --model_path ./checkpoint/model_best/checkpoint-1800/ \
    --test_path ../data/cluener/dev.txt \
    --batch_size 128 \
    --max_seq_len 64 \
    --debug

"""
[2022-12-22 06:06:36,674] [    INFO] - -----------------------------
[2022-12-22 06:06:36,674] [    INFO] - Class Name: 组织
[2022-12-22 06:06:36,674] [    INFO] - Evaluation Precision: 0.90691 | Recall: 0.82289 | F1: 0.86286
[2022-12-22 06:06:38,077] [    INFO] - -----------------------------
[2022-12-22 06:06:38,077] [    INFO] - Class Name: 地址
[2022-12-22 06:06:38,077] [    INFO] - Evaluation Precision: 0.82095 | Recall: 0.65147 | F1: 0.72646
[2022-12-22 06:06:39,865] [    INFO] - -----------------------------
[2022-12-22 06:06:39,865] [    INFO] - Class Name: 人名
[2022-12-22 06:06:39,866] [    INFO] - Evaluation Precision: 0.92936 | Recall: 0.90538 | F1: 0.91721
[2022-12-22 06:06:40,473] [    INFO] - -----------------------------
[2022-12-22 06:06:40,474] [    INFO] - Class Name: 书籍
[2022-12-22 06:06:40,474] [    INFO] - Evaluation Precision: 0.96774 | Recall: 0.77922 | F1: 0.86331
[2022-12-22 06:06:41,881] [    INFO] - -----------------------------
[2022-12-22 06:06:41,881] [    INFO] - Class Name: 公司
[2022-12-22 06:06:41,881] [    INFO] - Evaluation Precision: 0.91541 | Recall: 0.80159 | F1: 0.85472
[2022-12-22 06:06:42,411] [    INFO] - -----------------------------
[2022-12-22 06:06:42,411] [    INFO] - Class Name: 电影
[2022-12-22 06:06:42,411] [    INFO] - Evaluation Precision: 0.91339 | Recall: 0.76821 | F1: 0.83453
[2022-12-22 06:06:43,514] [    INFO] - -----------------------------
[2022-12-22 06:06:43,515] [    INFO] - Class Name: 游戏
[2022-12-22 06:06:43,515] [    INFO] - Evaluation Precision: 0.90508 | Recall: 0.90508 | F1: 0.90508
[2022-12-22 06:06:44,495] [    INFO] - -----------------------------
[2022-12-22 06:06:44,496] [    INFO] - Class Name: 政府
[2022-12-22 06:06:44,496] [    INFO] - Evaluation Precision: 0.93103 | Recall: 0.87449 | F1: 0.90188
[2022-12-22 06:06:46,315] [    INFO] - -----------------------------
[2022-12-22 06:06:46,315] [    INFO] - Class Name: 职位
[2022-12-22 06:06:46,315] [    INFO] - Evaluation Precision: 0.89975 | Recall: 0.82910 | F1: 0.86298
[2022-12-22 06:06:47,160] [    INFO] - -----------------------------
[2022-12-22 06:06:47,161] [    INFO] - Class Name: 景点
[2022-12-22 06:06:47,161] [    INFO] - Evaluation Precision: 0.85638 | Recall: 0.77033 | F1: 0.81108
"""
```

可自行上传到[CLUE2020榜单](https://www.cluebenchmarks.com/ner.html)进行测试。

预测：

```python
from pprint import pprint
from paddlenlp import Taskflow
import json
import random
cluener_en_label2_zh = {
  "organization": "组织",
  "name":"人名",
  "address":"地址",
  "company":"公司",
  "government":"政府",
  "book":"书籍",
  "game":"游戏",
  "movie":"电影",
  "position":"职位",
  "scene":"景点",
}
schema = list(cluener_en_label2_zh.values())
# 设定抽取目标和定制化模型权重路径
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best/checkpoint-1800/')
with open("../data/cluener/test.json", "r", encoding="utf-8") as fp:
  data = fp.read().strip().split("\n")

text = random.choice(data)
text = json.loads(text)["text"]
print(text)
pprint(my_ie(text))
print("="*100)
"""
死骑的大招是DOTA中屈指可数的不怕末日大招的技能。他的冰刀也让他有一定成为伪核的潜力。
[{'人名': [{'end': 2,
          'probability': 0.8145532957473662,
          'start': 0,
          'text': '死骑'}],
  '游戏': [{'end': 10,
          'probability': 0.9937154293496349,
          'start': 6,
          'text': 'DOTA'}]}]
====================================================================================================
"""
```

预测test.json并保存数据到txt中：

```python
python predict_to_file.py
```

# 数据蒸馏

text/process.py中的show_cluener_doccano_txt()用于统计转换后的test.txt里面实体的数目。

```python
+--------+------+
| 标签名 | 数目 |
+--------+------+
|  组织  | 321  |
|  人名  | 347  |
|  地址  | 303  |
|  公司  | 289  |
|  政府  | 166  |
|  书籍  | 103  |
|  游戏  | 219  |
|  电影  | 123  |
|  职位  | 314  |
|  景点  | 173  |
+--------+------+
```

这里数据蒸馏指的是首先利用少量标注的样本训练一个模型，然后再预测大量无标签的样本，最后将预测的结果加入到有标签的样本中继续训练。因此，这里选择test.txt作为有标签的样本，train_doccano.json里面的text作为未标注的数据，使用dev.txt作为验证数据。

进入到text下，根据有标签的数据训练一个模型：

```python
python finetune.py  \
    --device gpu \
    --logging_steps 100 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir /content/sample_data \
    --train_path ../data/cluener/test.txt \
    --dev_path ../data/cluener/dev.txt  \
    --max_seq_len 64  \
    --per_device_train_batch_size  128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir /content/sample_data/ \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1

```

验证：

```python
python evaluate.py \
    --model_path /content/sample_data/checkpoint-1680 \
    --test_path ../data/cluener/dev.txt \
    --batch_size 128 \
    --max_seq_len 64

"""[2022-12-22 08:57:37,182] [    INFO] - Evaluation Precision: 0.79103 | Recall: 0.75781 | F1: 0.77406"""

python evaluate.py \
    --model_path /content/sample_data/checkpoint-1680 \
    --test_path ../data/cluener/dev.txt \
    --batch_size 128 \
    --max_seq_len 64 \
    --debug

"""
[2022-12-22 08:58:46,785] [    INFO] - -----------------------------
[2022-12-22 08:58:46,785] [    INFO] - Class Name: 组织
[2022-12-22 08:58:46,785] [    INFO] - Evaluation Precision: 0.87387 | Recall: 0.79292 | F1: 0.83143
[2022-12-22 08:58:48,151] [    INFO] - -----------------------------
[2022-12-22 08:58:48,151] [    INFO] - Class Name: 地址
[2022-12-22 08:58:48,152] [    INFO] - Evaluation Precision: 0.83594 | Recall: 0.57373 | F1: 0.68045
[2022-12-22 08:58:49,913] [    INFO] - -----------------------------
[2022-12-22 08:58:49,913] [    INFO] - Class Name: 人名
[2022-12-22 08:58:49,913] [    INFO] - Evaluation Precision: 0.93039 | Recall: 0.86237 | F1: 0.89509
[2022-12-22 08:58:50,524] [    INFO] - -----------------------------
[2022-12-22 08:58:50,525] [    INFO] - Class Name: 书籍
[2022-12-22 08:58:50,525] [    INFO] - Evaluation Precision: 0.94393 | Recall: 0.65584 | F1: 0.77395
[2022-12-22 08:58:51,933] [    INFO] - -----------------------------
[2022-12-22 08:58:51,934] [    INFO] - Class Name: 公司
[2022-12-22 08:58:51,934] [    INFO] - Evaluation Precision: 0.90909 | Recall: 0.79365 | F1: 0.84746
[2022-12-22 08:58:52,448] [    INFO] - -----------------------------
[2022-12-22 08:58:52,448] [    INFO] - Class Name: 电影
[2022-12-22 08:58:52,448] [    INFO] - Evaluation Precision: 0.92000 | Recall: 0.76159 | F1: 0.83333
[2022-12-22 08:58:53,539] [    INFO] - -----------------------------
[2022-12-22 08:58:53,540] [    INFO] - Class Name: 游戏
[2022-12-22 08:58:53,540] [    INFO] - Evaluation Precision: 0.89310 | Recall: 0.87797 | F1: 0.88547
[2022-12-22 08:58:54,488] [    INFO] - -----------------------------
[2022-12-22 08:58:54,488] [    INFO] - Class Name: 政府
[2022-12-22 08:58:54,488] [    INFO] - Evaluation Precision: 0.91220 | Recall: 0.75709 | F1: 0.82743
[2022-12-22 08:58:56,222] [    INFO] - -----------------------------
[2022-12-22 08:58:56,222] [    INFO] - Class Name: 职位
[2022-12-22 08:58:56,222] [    INFO] - Evaluation Precision: 0.86352 | Recall: 0.75982 | F1: 0.80835
[2022-12-22 08:58:56,853] [    INFO] - -----------------------------
[2022-12-22 08:58:56,853] [    INFO] - Class Name: 景点
[2022-12-22 08:58:56,854] [    INFO] - Evaluation Precision: 0.78443 | Recall: 0.62679 | F1: 0.69681
"""
```

进入到data_distill下，先获取蒸馏数据，运行：

```python
python data_distill.py \
    --data_path /content/drive/MyDrive/project/FewNer/data/cluener/\
    --save_dir /content/drive/MyDrive/project/FewNer/data/student_data \
    --task_type entity_extraction \
    --synthetic_ratio 10 \
    --platform doccano \
    --model_path /content/drive/MyDrive/project/FewNer/checkpoint-1680

"""
[2022-12-22 14:57:07,879] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load '/content/drive/MyDrive/project/FewNer/checkpoint-1680'.
[2022-12-22 15:50:49,221] [    INFO] - Save 10748 examples to /content/drive/MyDrive/project/FewNer/data/student_data/train_data.json.
[2022-12-22 15:50:49,970] [    INFO] - Save 1343 examples to /content/drive/MyDrive/project/FewNer/data/student_data/dev_data.json.
"""
```

评估教师模型：

```python
python evaluate_teacher.py \
    --task_type entity_extraction \
    --test_path /content/drive/MyDrive/project/FewNer/data/student_data/dev_data.json \
    --label_maps_path /content/drive/MyDrive/project/FewNer/data/student_data/label_maps.json \
    --model_path /content/drive/MyDrive/project/FewNer/checkpoint-1680 \
    --batch_size 128 \
    --max_seq_len 64

"""
[2022-12-22 10:47:52,620] [    INFO] - Evaluation precision: {'entity_f1': 0.77394, 'entity_precision': 0.79076, 'entity_recall': 0.75781}
"""
```

根据预测的数据训练模型：

```python
python train.py \
    --task_type entity_extraction \
    --train_path /content/drive/MyDrive/project/FewNer/data/student_data/train_data.json \
    --dev_path /content/drive/MyDrive/project/FewNer/data/student_data/dev_data.json \
    --label_maps_path /content/drive/MyDrive/project/FewNer/data/student_data/label_maps.json \
    --num_epochs 50 \
    --encoder ernie-3.0-mini-zh \
    --batch_size 128 \
    --max_seq_len 64 \
    --device gpu \
    --logging_steps 10 \
    --eval_steps 200 \
    --save_dir /content/sample_data/
```

评估模型：

```python
python evaluate.py \
    --model_path /content/sample_data/model_best \
    --test_path /content/drive/MyDrive/project/FewNer/data/student_data/dev_data.json \
    --task_type entity_extraction \
    --label_maps_path /content/drive/MyDrive/project/FewNer/data/student_data/label_maps.json \
    --encoder ernie-3.0-mini-zh

"""
Evaluation precision: {'entity_f1': 0.76643, 'entity_precision': 0.80335, 'entity_recall': 0.73275}
"""
```

发现并没有比之前的好，可能的原因是在**继续训练的时候，没有加入之前有标注的数据，而是直接采用预测的数据。**

使用模型进行预测：

```python
from pprint import pprint
from paddlenlp import Taskflow

my_ie = Taskflow("information_extraction", model="uie-data-distill-gp", task_path="/content/sample_data/model_best") # Schema is fixed in closed-domain information extraction
pprint(my_ie("威尔哥（Virgo）减速炸弹是由瑞典FFV军械公司专门为瑞典皇家空军的攻击机实施低空高速轰炸而研制，1956年开始研制，1963年进入服役，装备于A32“矛盾”、A35“龙”、和AJ134“雷”攻击机，主要用于攻击登陆艇、停放的飞机、高炮、野战火炮、轻型防护装甲车辆以及有生力量。"))

"""
[{'公司': [{'end': 25,
          'probability': 0.9994653,
          'start': 16,
          'text': '瑞典FFV军械公司'}],
  '政府': [{'end': 34, 'probability': 0.9953176, 'start': 28, 'text': '瑞典皇家空军'}]}]
"""
```

# 参考

> [CLUEbenchmark/CLUENER2020: CLUENER2020 中文细粒度命名实体识别 Fine Grained Named Entity Recognition (github.com)](https://github.com/CLUEbenchmark/CLUENER2020)
>
> [PaddleNLP/applications/information_extraction/text at develop · PaddlePaddle/PaddleNLP (github.com)](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/information_extraction/text)
