import json
from prettytable import PrettyTable
from collections import defaultdict

def process_cluener_main():
  """该函数主要是将数据处理成doccano标注好的格式"""
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

  def process_cluener(path, out_path=None):
    label_count = {k:0 for k in cluener_en_label2_zh.values()}
    length_count = defaultdict(int)
    with open(path, "r", encoding="utf-8") as fp:
      data = fp.read().strip().split("\n")

    ofp = open(out_path, "w", encoding="utf-8")
    for i,d in enumerate(data):
      d = json.loads(d)
      label = d["label"]
      target = {}
      target["id"] = i
      target["text"] = d["text"]
      length_count[len(d["text"])] += 1
      target["relations"] = []
      target["entities"] = []
      tmp_res = []
      for k,v in label.items():
        label_count[cluener_en_label2_zh[k.lower()]] += len(v)
        for kk, vv in v.items():
          for vvv in vv:
            tmp = {}
            tmp["label"] = cluener_en_label2_zh[k]
            tmp["start_offset"] = vvv[0]
            tmp["end_offset"] = vvv[1] + 1  # 实体的后一位
            tmp_res.append(tmp)
      tmp_res = sorted(tmp_res, key=lambda x:x["start_offset"])
      tmp_res = [{"id":i, "label":v["label"], "start_offset":v["start_offset"], "end_offset":v["end_offset"]} for i,v in enumerate(tmp_res)]
      target["entities"] = tmp_res
      ofp.write(json.dumps(target, ensure_ascii=False) + "\n")


    ## 按行添加数据
    tb = PrettyTable()
    tb.field_names = ["标签名", "数目"]
    for k,v in label_count.items():
      tb.add_row([k, v])
    print(tb)

    ## 按行添加数据
    tb2 = PrettyTable()
    tb2.field_names = ["文本长度", "数目"]
    length_count = sorted(length_count.items(), key=lambda x:x[0])
    for k,v in length_count:
      tb2.add_row([k, v])
    print(tb2)
    print("="*100)

  process_cluener("../data/cluener/train.json", "../data/cluener/train_doccano.json")
  process_cluener("../data/cluener/dev.json", "../data/cluener/dev_doccano.json")

def show_cluener_doccano_txt():
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
  label_count = {k:0 for k in cluener_en_label2_zh.values()}
  with open("../data/cluener/test.txt", "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
  for d in data:
    d = json.loads(d)
    label_count[d["prompt"]] += len(d["result_list"])
  ## 按行添加数据
  tb = PrettyTable()
  tb.field_names = ["标签名", "数目"]
  for k,v in label_count.items():
    tb.add_row([k, v])
  print(tb)

if __name__ == "__main__":
  # process_cluener_main()
  show_cluener_doccano_txt()
