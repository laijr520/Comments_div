import pandas as pd

df = pd.read_csv('comments\comments_1.csv')  # 改成你的文件名

# 拼接父评论作为上下文
id2content = dict(zip(df['评论ID'].astype('int64'), df['评论内容'].astype(str)))

def build_text(row):
    content = str(row['评论内容']).strip()
    if pd.notna(row['上级评论ID']):
        parent = id2content.get(int(row['上级评论ID']), '')
        if parent:
            return f"[父评论] {parent}\n[回复] {content}"
    return content

df['待分析文本'] = df.apply(build_text, axis=1)

# 清洗明显噪声：过短的表情/单字直接标记为 none，不送API省钱
df['预标记'] = df['评论内容'].astype(str).apply(
    lambda x: 'none' if len(x.strip()) <= 3 or x.strip().startswith('[') and x.strip().endswith(']') else ''
)

df[['评论ID', '评论内容', '待分析文本', '预标记']].to_csv(
    'step1_prepared.csv', index=False, encoding='utf-8-sig'
)
print(f'总数 {len(df)}, 需调API {(df["预标记"]=="").sum()} 条')