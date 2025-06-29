import functools
import json
import torch
import pandas as pd
from tqdm import tqdm
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

NEW_MAX_TOKENS = 3072

model_name="/home/whu/qwen3/model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def call_gpt_local(prompt: str, max_new_tokens: int = NEW_MAX_TOKENS):
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # 启用思考模式
    )

    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 模型生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    print(thinking_content)
    print(content)
    return content


def call_gpt_local_stream(prompt: str, max_new_tokens: int = NEW_MAX_TOKENS):
    messages = [{"role": "user", "content": prompt}]

    # 构建输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 创建 Streamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # 推理线程
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 实时输出内容
    full_text = ""
    print(">>> Streaming:")
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_text += new_text

    print()  # 结尾换行
    return full_text.strip()

def extract_prompt(text, entity_list):
    if not text:
        return "输入文本不能为空"
    if not isinstance(entity_list, list) or not all(isinstance(item, str) for item in entity_list):
        return "指定实体列表应为字符串列表"

    prompt_template = """请从以下文本中提取“实体”及其“属性”，并严格按照指定的 JSON 格式输出。

【输入文本】
%s

【指定实体列表】
%s

【生成要求】
1. 仅从输入文本中提取实体及其属性，文本无实体，禁止胡编乱造;
2. 若文本中出现指定实体列表的实体，提取这些指定实体和其他实体；
3. 若文本中未出现任何指定实体列表的实体，仅提取“其他实体”及其属性；
4. 严格按照指定的 JSON 格式输出，不要添加任何解释、注释或额外说明；
5. 属性值尽量提取原文表达，保持简洁、准确；
6. 实体必须是中文名词，禁止把数字、英文视为实体；
7. 属性必须是实体的性质，必须是名词；
8. 实体数量不得超过10个，禁止过度抽取，禁止实体超过10个！！！！

【输出格式】
[
  {
    "实体名称": "实体在文本中的名称",
    "属性": {
      "属性名1": "属性值1",
      "属性名2": "属性值2",
      ...
    },...
  }
]

现在开始提取，请直接输出符合要求的 JSON,实体不能超过10个：
"""
    prompt = prompt_template % (str(text), str(entity_list))
    return prompt

def write_data_json(data_list):
    with open("output2.json","a",encoding="utf-8") as f:
        for e in data_list:
            json_str = json.dumps(e,ensure_ascii=False)
            f.write(json_str+"\n")
    print("save!!!")


def split_list(lst):
    size = len(lst)//50
    remain =len(lst)%50

    total_lst = []
    for i in range(size):
        ls=lst[i*50:i*50+50]
        total_lst.append(ls)

    if remain:
        ls=lst[-remain:]
        total_lst.append(ls)
    return total_lst



def read_txt(path):
    with open(path,"r",encoding="utf-8") as f:
        data_list = f.readlines()
    return [e.strip() for e in data_list if e.strip()]


def run(excel_file,entity_path,output_file="entity_extract.xlsx"):
    df = pd.read_excel(excel_file)
    entity_list = read_txt(entity_path)

    count = 0
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        count +=1
        if count<=68:
            continue
        try:
            text = row['text']
            print("raw data\n",text)
            prompt = extract_prompt(text, entity_list)
            response_json = call_gpt_local_stream(prompt)
            try:
                try:
                    parsed = json.loads(response_json)
                except:
                    parsed = eval(response_json)
            except:
                parsed = {"error": "Invalid JSON"}

            entity_candidate =[]
            entity_other = []
            for item in parsed:
                name = item["实体名称"]
                if name in entity_list:
                    entity_candidate.append(name)
                else:
                    entity_other.append(name)
            one_data = {
                "id": row["id"],
                "chunk_id": row["chunk_id"],
                "text": text,
                "extracted": parsed,
                "entity_candidate":entity_candidate,
                "entity_other":entity_other,
            }

            results.append(one_data)
            write_data_json([one_data])
        except:
            continue
    output_df = pd.DataFrame(results)
    output_df.to_excel(output_file, index=False)
    print(f"✅ 提取完成，结果已保存到 {output_file}")


if __name__ == '__main__':
    excel_file, entity_path ='/home/whu/workspace/knowledge_graph-main/book.xlsx','/home/whu/workspace/knowledge_graph-main/实体.txt'
    run(excel_file,entity_path)





