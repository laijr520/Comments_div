import os, json, time
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    api_key="sk-38b99775a32c4c48b0a1729ce9edb1cc",
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """你是社会心理学研究助手，对中文足球评论做认知偏误分类。

【六类别】
1. confirmation_bias 确认偏误：只选择性记对己方不利判罚，无视对己方有利的；预设结论当事实
2. ingroup_bias 组内偏见：美化己方球员/球迷/球队，贬低对方整体
3. fundamental_attribution_error 基本归因错误：对方犯错归于人品/本性，己方同样行为归于情境/战术
4. motivated_reasoning 动机性推理：面对回放/规则/数据仍用扭曲逻辑推出符合己方立场结论
5. hostile_attribution_bias 敌意归因偏差：把误判、赛程、庆祝解读为蓄意针对、剧本、阴谋
6. none 无明显偏误或无法判断：纯玩梗/表情/口号/对话片段/理性分析/上下文不足

【输出格式】严格JSON：
{"primary":"类别英文名","secondary":"类别英文名或null","confidence":0.0~1.0,"reason":"不超过30字"}

【规则】
- 纯表情、梗、单字附和 → primary=none
- 回复评论脱离上下文无法判断 → primary=none, reason="上下文不足"
- 引用规则/反事实推理/承认己方错误 → primary=none
- 仅当确有两种偏误同时存在时填 secondary
"""

RESULT_FILE = 'step2_results.jsonl'

def classify_one(cid, text, pre_tag):
    if pre_tag == 'none':
        return {"评论ID": int(cid), "primary": "none", "secondary": None,
                "confidence": 1.0, "reason": "预过滤：表情/过短"}
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"分析这条评论：\n\n{text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,   # 分类任务用0保证一致性
            max_tokens=200,
        )
        out = json.loads(resp.choices[0].message.content)
        out["评论ID"] = int(cid)
        return out
    except Exception as e:
        return {"评论ID": int(cid), "primary": "ERROR", "secondary": None,
                "confidence": 0.0, "reason": f"API错误: {str(e)[:50]}"}

def main():
    df = pd.read_csv('step1_prepared.csv')
    # 断点续跑
    done = set()
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, encoding='utf-8') as f:
            done = {json.loads(l)['评论ID'] for l in f}
        print(f'已完成 {len(done)} 条，跳过')
    todo = df[~df['评论ID'].isin(done)]
    print(f'本次处理 {len(todo)} 条')

    with open(RESULT_FILE, 'a', encoding='utf-8') as fout, \
         ThreadPoolExecutor(max_workers=8) as ex:   # 8并发，对DeepSeek很友好
        futures = {
            ex.submit(classify_one, r['评论ID'], r['待分析文本'], r['预标记']): r['评论ID']
            for _, r in todo.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()
            if i % 20 == 0:
                print(f'  {i}/{len(todo)}')

if __name__ == '__main__':
    main()