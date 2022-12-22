# 查找重复出现的字符串
def str_all_index(str_, a):
    '''
    Parameters
    ----------
    str_ : string.
    a : str_中的子串

    Returns
    -------
    index_list : list

    首先输入变量2个，输出list，然后中间构造每次find的起始位置start,start每次都在找到的索引+1，后面还得有终止循环的条件

    '''
    index_list=[]
    start=0
    while True:
        x=str_.find(a,start)
        if x>-1:
            start=x+1
            index_list.append(x)
        else:
            break
    return index_list

# 预测
import json
from paddlenlp import Taskflow

def predict(ie, text):
    result = ie(text)[0]
    # print(result)
    sub_list = []
    for item in result:
        sub_tmp = {}
        sub_tmp["label"] = item
        all_search=str_all_index(text, result[item][0]["text"])
        txt_list=[]
        sub_span=[]
        if len(result[item])==1:
            for i in range(len(all_search)):
                txt_list.append(result[item][0]["text"])
                sub_span.append([all_search[i], all_search[i] +len(result[item][0]["text"])])
        else:
            for i in range(len(result[item])):
                txt_list.append(result[item][i]["text"])
                sub_span.append([result[item][i]["start"], result[item][i]["end"]])
        sub_tmp["text"]=txt_list
        sub_tmp["span"]=sub_span
        sub_list.append(sub_tmp)
    result_item["entities"] = sub_list
    return result_item

# 设定抽取目标和定制化模型权重路径
ie = Taskflow("information_extraction", schema=schema, batch_size=32, precision='fp32', use_faster=True, task_path='./checkpoint/model_best/checkpoint-1800/')

ff = open('../data/output/cluener_result_base.txt', 'w')
with open("../data/cluener/test.json", "r", encoding="utf-8") as fp:
  data = fp.read().strip().split("\n")
for line in data:
  result_item=json.loads(line)
  text = result_item["text"]
  target = predict(ie, text)
  ff.write(json.dumps(target, ensure_ascii=False) + '\n')
ff.close()
print("数据结果已导出")