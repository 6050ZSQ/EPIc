import json
import requests
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time  # 用于设置重试的时间间隔

url = "https://api.bltcy.cn/v1/chat/completions"
model_list = [
    # 'deepseek-r1',
    # 'deepseek-v3',
    # 'gpt-4o-mini-2024-07-18',
    # 'o1-mini-2024-09-12',
    # 'gemini-2.5-pro-exp-03-25',
    'gemini-2.5-flash-preview-04-17',
]
for model in model_list:
    json_data = {}

    sheet_list = [
        # 'presupposition', 
        # 'entailment', 
        'implicature'
    ]

    prompt_list = [
        '你现在是一个中文母语者。对于以下已知情况，在给出的四个选项中，选择你认为符合已知情况的选项，允许多选或不选。\n 已知：\"{Premise}\" \n \"{Hypothesis}\" \n 回答格式为\"Reason: , Answer: "',
    ]

    for sheet, prompt in zip(sheet_list, prompt_list):
        true_num = 0
        json_data[sheet] = []
        print(f"\n正在读取 sheet: {sheet}")
        # 读取当前 sheet 的数据
        file_path = f'{sheet}.csv'  # 文件路径
        data = pd.read_csv(file_path, sheet_name=sheet)

        for i in tqdm(range(120)):
            new_data = {}

            row_number = i
            selected_row = data.iloc[row_number]
            number = selected_row['Number']
            type = selected_row['Type']
            is_nature = selected_row['Nature/AI']
            premise = selected_row['Premise']
            hypothesis = selected_row['Hypothesis']
            res = selected_row['Res']

            headers = {
                'Accept': 'application/json',
                'Authorization': 'Bearer YOUR API KEY',
                'Content-Type': 'application/json'
            }

            payload = json.dumps({
                "model": f"{model}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt.format(Premise=premise, Hypothesis=hypothesis)
                    }
                ]
            })

            # 添加重试机制
            max_retries = 3  # 最大重试次数
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    response.raise_for_status()  # 如果响应状态码不是 200，会抛出异常
                    response = json.loads(response.text)
                    
                    # 检查响应中是否有所需的内容
                    if "choices" in response and len(response["choices"]) > 0:
                        success = True  # 请求成功，退出重试循环
                    else:
                        raise ValueError("Response does not contain valid choices.")
                except Exception as e:
                    retry_count += 1
                    print(f"请求失败，正在重试 {retry_count}/{max_retries} ... 错误信息: {e}")
                    time.sleep(2)  # 等待 2 秒后重试

            if not success:
                print(f"请求失败，跳过此条数据: Number={number}")
                new_data['Number'] = int(number)
                new_data['Type'] = type
                new_data['Nature/AI'] = is_nature
                new_data['Premise'] = premise
                new_data['Hypothesis'] = hypothesis
                new_data['Res'] = res
                new_data['answer'] = "N/A"
                new_data['response'] = "Failed after retries"
                json_data[sheet].append(new_data)
                continue

            answer = response["choices"][0]["message"]["content"].split('Answer:')[-1]
            if res in answer:
                true_num += 1
            
            new_data['Number'] = int(number)
            new_data['Type'] = type
            new_data['Nature/AI'] = is_nature
            new_data['Premise'] = premise
            new_data['Hypothesis'] = hypothesis
            new_data['Res'] = res
            new_data['answer'] = answer
            new_data['response'] = response["choices"][0]["message"]["content"]

            json_data[sheet].append(new_data)

    with open(f'{model}_0.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
