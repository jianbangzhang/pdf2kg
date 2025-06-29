import functools
import json
import torch
import pandas as pd
from tqdm import tqdm







def extract_prompt(entity, content_list):
    if not entity:
        return "输入文本不能为空"
    if not isinstance(content_list, list) or not all(isinstance(item, str) for item in content_list):
        return "指定实体列表应为字符串列表"

    prompt_template = f"""请从以下文本中提取实体{entity}的属性，并严格按照指定的 JSON 格式输出。
【输入文本】
{" ".join(content_list)}
【实体】
{entity}

【生成要求】
1. 仅从输入文本中提取实体{entity}的属性，禁止胡编乱造;
2. 严格按照指定的 JSON 格式输出，不要添加任何解释、注释或额外说明；
3. 属性值尽量提取原文表达，保持简洁、准确；
4. 属性必须是实体的性质，必须是名词；
5. 只抽取{str(entity)}的属性，禁止抽取其他的实体，即非{str(entity)}的实体禁止抽取；
6. 属性名和属性值，禁止混为一谈，如无属性值可不填；
7. 相同含义的属性名，禁止重复！

【输出格式】
{{"实体名称": f"{str(entity)}","属性": {{"属性名1": "属性值1","属性名2": "属性值2",...}}}}

现在开始提取{entity}的属性，请直接输出符合要求的 JSON：
"""
    return prompt_template


def write_data_json(data_list):
    with open("output_entity.json","a",encoding="utf-8") as f:
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


from openai import OpenAI


def call_ds(user_message, api_key="xxxxx", system_message="你是实体属性抽取专家", model="deepseek-chat",stream=True):
    """
    调用 DeepSeek API 获取聊天回复

    参数:
        api_key (str): DeepSeek API 密钥
        user_message (str): 用户输入的消息
        system_message (str, optional): 系统角色消息，默认为"You are a helpful assistant"
        model (str, optional): 使用的模型，默认为"deepseek-chat"
        stream (bool, optional): 是否使用流式响应，默认为False

    返回:
        str: 模型的回复内容
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            stream=stream
        )

        if stream:
            # 处理流式响应
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            return full_response
        else:
            # 处理非流式响应
            return response.choices[0].message.content

    except Exception as e:
        return f"调用API时出错: {str(e)}"


def read_txt(path):
    with open(path,"r",encoding="utf-8") as f:
        data_list = f.readlines()
    return [e.strip() for e in data_list if e.strip()]


def run(excel_file,entity_path,output_file="entity.xlsx"):
    df = pd.read_excel(excel_file)
    entity_list = read_txt(entity_path)
    results = []
    text_list = df["text"]

    for i,e in enumerate(entity_list):
        content_list = []
        for text in text_list:
            if e in text:
                content_list.append(text)
            else:
                continue
        if not content_list:
            continue

        prompt = extract_prompt(e,content_list)
        response_json = call_ds(prompt)
        print("data id",i)
        print("entity:",e)
        print("result\n",response_json)
        try:
            try:
                parsed = json.loads(response_json)
            except:
                parsed = eval(response_json)
        except:
            parsed = {"error": "Invalid JSON"}

        one_data = {
            "text": content_list,
            "entity": e,
            "extracted": parsed,
        }

        results.append(one_data)
        write_data_json([one_data])
    output_df = pd.DataFrame(results)
    output_df.to_excel(output_file, index=False)
    print(f"✅ 提取完成，结果已保存到 {output_file}")


if __name__ == '__main__':
    excel_file, entity_path ='/home/whu/workspace/knowledge_graph-main/book.xlsx','/home/whu/workspace/knowledge_graph-main/实体.txt'
    run(excel_file,entity_path)
