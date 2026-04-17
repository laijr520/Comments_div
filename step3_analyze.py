import json, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 合并
results = [json.loads(l) for l in open('step2_results.jsonl', encoding='utf-8')]
res_df = pd.DataFrame(results)
raw_df = pd.read_csv('step1_prepared.csv')
merged = raw_df.merge(res_df, on='评论ID', how='left')
merged.to_csv('final_labeled.csv', index=False, encoding='utf-8-sig')

# 分布统计
label_map = {
    'confirmation_bias':'确认偏误','ingroup_bias':'组内偏见',
    'fundamental_attribution_error':'基本归因错误','motivated_reasoning':'动机性推理',
    'hostile_attribution_bias':'敌意归因偏差','none':'无明显偏误','ERROR':'调用失败'
}
merged['主分类'] = merged['primary'].map(label_map).fillna('其他')

counts = merged['主分类'].value_counts()
print('\n=== 主偏误分布 ===')
print(counts)
print(f'\n有明显偏误占比: {(merged["primary"]!="none").mean():.1%}')

# 饼图
fig, ax = plt.subplots(figsize=(9, 7))
ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title('B站评论认知偏误分类分布 (n=%d)' % len(merged))
plt.tight_layout()
plt.savefig('fig_pie.png', dpi=150)

# 偏误类型内部对比（排除none）
bias_only = merged[~merged['primary'].isin(['none','ERROR'])]
bias_counts = bias_only['主分类'].value_counts()
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(bias_counts.index, bias_counts.values)
ax2.set_ylabel('评论数')
ax2.set_title('各类认知偏误出现频次 (n=%d)' % len(bias_only))
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('fig_bar.png', dpi=150)

# 分层抽样：每类最多取20条
import numpy as np
np.random.seed(42)

samples = []
for cat, group in merged.groupby('primary'):
    n = min(20, len(group))
    samples.append(group.sample(n, random_state=42))
sample = pd.concat(samples, ignore_index=True)

# 加一列空的"人工标签"方便你后面填
sample['人工标签'] = ''

sample[['评论ID','评论内容','primary','secondary','confidence','reason','人工标签']].to_csv(
    'review_sample.csv', index=False, encoding='utf-8-sig'
)
print(f'\n人工复核样本已导出: review_sample.csv (共{len(sample)}条)')