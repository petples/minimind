"""
Build Brain Trading SFT dataset from PPO backtest case files.
v2: Massive response diversity — 17 templates to prevent repetition loops.
"""

import json
import random
import os
import numpy as np

random.seed(42)
np.random.seed(42)

INPUT_PATH = "/root/brain/data/ppo_cases_US_SPY.json"
OUTPUT_PATH = "/root/minimind/dataset/brain_trading_sft.jsonl"
MAX_SAMPLES = 3000

SYSTEM_PROMPT = "你是一个专业的量化交易分析师。根据提供的市场数据，判断应该做多还是做空，并给出简要理由。"

def derive_named_features(feat_vals):
    price_momentum = feat_vals[0]
    order_flow = feat_vals[1]
    micro_dev = feat_vals[2]
    vol_proxy = feat_vals[4]
    liq_pressure = feat_vals[11]
    price = 4500.0 + price_momentum * 200.0
    price = round(price, 2)
    rsi = 50.0 + price_momentum * 35.0
    rsi = round(np.clip(rsi, 25, 75), 1)
    vpin = 0.35 + (order_flow * 0.3) + (liq_pressure * 0.15)
    vpin = round(np.clip(vpin, 0.15, 0.85), 2)
    vol_20 = 0.10 + (vol_proxy - 3.4) / (11.6 - 3.4) * 0.25
    vol_20 = round(np.clip(vol_20, 0.08, 0.40), 3)
    turtle = 0.3 + price_momentum * 0.4 + order_flow * 0.2
    turtle = round(np.clip(turtle, 0.05, 0.95), 2)
    ob_imbalance = order_flow * 0.8
    ob_imbalance = round(np.clip(ob_imbalance, -0.70, 0.70), 2)
    frac_diff = price_momentum * 0.04 + np.random.normal(0, 0.005)
    frac_diff = round(np.clip(frac_diff, -0.08, 0.08), 4)
    clenow = 0.5 + price_momentum * 0.3 + (vol_proxy - 5) / 10.0 * 0.2
    clenow = round(np.clip(clenow, 0.30, 0.95), 2)
    sakata = 0.5 + order_flow * 0.35 + liq_pressure * 0.15
    sakata = round(np.clip(sakata, 0.20, 0.95), 2)
    return {"price":price,"rsi":rsi,"vpin":vpin,"volatility_20":vol_20,
            "turtle_channel_position":turtle,"order_book_imbalance":ob_imbalance,
            "fractional_diff":frac_diff,"clenow_quality":clenow,"sakata_intensity":sakata}

def build_user_prompt(feats):
    parts = [f"SPX 当前价格 {feats['price']}", f"RSI {feats['rsi']}", f"VPIN {feats['vpin']}",
             f"波动率(20日) {feats['volatility_20']}", f"分數差分(d=0.4) {feats['fractional_diff']}",
             f"海龟通道位置 {feats['turtle_channel_position']}", f"订单簿不平衡 {feats['order_book_imbalance']}",
             f"Clenow质量 {feats['clenow_quality']}", f"Sakata强度 {feats['sakata_intensity']}"]
    return "，".join(parts)

def build_reasons(feats):
    reasons = []
    vpin = feats['vpin']
    if vpin > 0.55: reasons.append(f"VPIN {vpin} 偏高，存在逆向选择风险")
    elif vpin < 0.30: reasons.append(f"VPIN {vpin} 较低，信息不对称风险可控")
    else: reasons.append(f"VPIN {vpin} 处于中等水平")
    rsi = feats['rsi']
    if rsi > 65: reasons.append(f"RSI {rsi} 处于超买区域")
    elif rsi < 35: reasons.append(f"RSI {rsi} 处于超卖区域")
    elif rsi > 55: reasons.append(f"RSI {rsi} 偏多")
    elif rsi < 45: reasons.append(f"RSI {rsi} 偏空")
    else: reasons.append(f"RSI {rsi} 中性")
    turtle = feats['turtle_channel_position']
    if turtle > 0.80: reasons.append(f"海龟通道位置 {turtle} 接近上轨，存在回调压力")
    elif turtle < 0.20: reasons.append(f"海龟通道位置 {turtle} 接近下轨，存在反弹机会")
    ob = feats['order_book_imbalance']
    if abs(ob) > 0.3:
        dw = "买方" if ob > 0 else "卖方"
        reasons.append(f"订单簿不平衡 {ob} 显示{dw}压力")
    cq = feats['clenow_quality']
    if cq > 0.75: reasons.append(f"Clenow质量 {cq}，趋势质量良好")
    elif cq < 0.45: reasons.append(f"Clenow质量 {cq} 偏低，趋势可靠性不足")
    si = feats['sakata_intensity']
    if si > 0.75: reasons.append(f"Sakata强度 {si}，动量充足")
    elif si < 0.40: reasons.append(f"Sakata强度 {si} 偏弱，动量不足")
    vol = feats['volatility_20']
    if vol > 0.25: reasons.append(f"波动率 {vol} 较高，需注意仓位管理")
    elif vol < 0.12: reasons.append(f"波动率 {vol} 偏低，市场较为平稳")
    return reasons

def build_assistant_response(side, feats):
    direction = "做多" if side == "long" else "做空"
    reasons = build_reasons(feats)
    if not reasons: return "当前数据不足以做出明确判断，建议观望。"
    n = min(len(reasons), random.randint(2, 5))
    selected = random.sample(reasons, n)
    t = random.randint(0, 16)
    if t == 0: return f"建议{direction}。{'；'.join(selected[:random.randint(1,3)])}。"
    if t == 1: return f"综合判断：建议{direction}。\n" + "\n".join(f"- {r}" for r in selected[:3])
    if t == 2: return f"从各项技术指标来看，{'，'.join(selected[:3])}。因此建议{direction}。"
    if t == 3: return f"信号明确，{direction}。{selected[0]}。"
    if t == 4: return f"倾向于{direction}，{'；'.join(selected[:2])}。但需注意市场可能反向波动，建议控制仓位。"
    if t == 5: return f"数据面：{'，'.join(selected[:3])}。综合来看建议{direction}，维持中等仓位。"
    if t == 6: return f"问：当前应如何操作？答：建议{direction}。{'；'.join(selected[:2])}。"
    if t == 7: return f"分析摘要：\n{'& '.join(selected[:3])}\n→ 结论：建议{direction}"
    if t == 8: return f"信号偏{direction}，但整体强度一般。{'，'.join(selected[:2])}。建议轻仓{direction}，设好止损。"
    if t == 9: return f" {'，'.join(selected[:3])}。综合以上指标，建议{direction}。"
    if t == 10:
        risk = "高" if feats['volatility_20'] > 0.2 else "中等"
        return f"判断为{direction}机会。{'；'.join(selected[:2])}。风险等级{risk}。"
    if t == 11:
        if random.random() < 0.4: return f"当前信号矛盾：{'，'.join(selected[:3])}。建议暂时观望，等待更明确信号。"
        return f"建议{direction}。{'；'.join(selected[:1])}。"
    if t == 12: return f"市场呈现{random.choice(['震荡','趋势','盘整'])}格局。{'，'.join(selected[:3])}。操作上建议{direction}。"
    if t == 13:
        bulls = [r for r in selected if any(w in r for w in ['超卖','偏多','反弹','买方','动量充足'])]
        bears = [r for r in selected if any(w in r for w in ['超买','偏空','回调','卖方','风险','不足'])]
        if bulls: return f"利多因素：{'，'.join(bulls[:2])}。建议{direction}。"
        if bears: return f"利空因素：{'，'.join(bears[:2])}。建议{direction}。"
        return f"综合分析，建议{direction}。{'；'.join(selected[:2])}。"
    if t == 14: return f"{direction}。{'，'.join(selected[:1])}。"
    if t == 15: return f"经过技术指标综合分析：\n1) {'，'.join(selected[:2])}；\n2) {'，'.join(selected[2:4]) if len(selected)>2 else ''}。\n结论：建议{direction}。"
    if t == 16:
        price = feats['price']
        sl = round(price*0.98,1) if side=='long' else round(price*1.02,1)
        tp = round(price*1.04,1) if side=='long' else round(price*0.96,1)
        return f"建议{direction}，入场{price}附近，止损{sl}，目标{tp}。理由：{'，'.join(selected[:2])}。"
    return f"建议{direction}。{'；'.join(selected[:2])}。"

def load_cases():
    with open(INPUT_PATH, 'r') as f: data = json.load(f)
    cases = []
    for c in data:
        feat_str = c['state_features']['features']
        feat_vals = [float(x) for x in feat_str.replace('[','').replace(']','').replace('\n',' ').split() if x]
        cases.append({"features_raw":feat_vals,"action_side":c['action']['side'],
                      "action_size":c['action']['size'],"reward":c['outcome']['reward'],
                      "step":c['outcome']['step'],"drawdown":c['outcome']['drawdown'],
                      "total_cost":c['outcome']['total_cost'],"llm_annotation":c.get('llm_annotation')})
    return cases

def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} cases")
    wins = [c for c in cases if c['reward'] > 0]
    neutral = [c for c in cases if c['reward'] == 0]
    print(f"Wins: {len(wins)}, Neutral: {len(neutral)}")
    samples = []
    for case in wins:
        feats = derive_named_features(case['features_raw'])
        samples.append({"conversations":[{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":build_user_prompt(feats)},
            {"role":"assistant","content":build_assistant_response('long',feats)}]})
    random.shuffle(neutral)
    for case in neutral[:800]:
        feats = derive_named_features(case['features_raw'])
        samples.append({"conversations":[{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":build_user_prompt(feats)},
            {"role":"assistant","content":build_assistant_response('long',feats)}]})
    print(f"Long samples: {len(samples)}")
    pool = cases.copy()
    random.shuffle(pool)
    short_needed = max(min(MAX_SAMPLES-len(samples),1200),1000)
    for i in range(short_needed):
        bc = pool[i%len(pool)]
        p = bc['features_raw'].copy()
        p[0]=np.clip(-p[0]+random.uniform(-.2,.2),-1.1,.9)
        p[1]=np.clip(-p[1]+random.uniform(-.15,.15),-.9,.5)
        p[2]=np.clip(-p[2]+random.uniform(-.1,.1),-.8,.3)
        p[11]=np.clip(-p[11]+random.uniform(-.1,.3),-1.3,.3)
        feats = derive_named_features(p)
        samples.append({"conversations":[{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":build_user_prompt(feats)},
            {"role":"assistant","content":build_assistant_response('short',feats)}]})
    print(f"Short samples: {short_needed}")
    remaining = MAX_SAMPLES-len(samples)
    for i in range(remaining):
        bc = random.choice(pool)
        p = bc['features_raw'].copy()
        p[0]=np.clip(p[0]+random.uniform(-.3,.3),-1.1,.9)
        p[1]=np.clip(p[1]+random.uniform(-.2,.2),-.9,.9)
        p[2]=np.clip(p[2]+random.uniform(-.15,.15),-.8,.8)
        p[11]=np.clip(p[11]+random.uniform(-.2,.2),-1.3,1.3)
        feats = derive_named_features(p)
        samples.append({"conversations":[{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":build_user_prompt(feats)},
            {"role":"assistant","content":build_assistant_response('long',feats)}]})
    random.shuffle(samples)
    samples = samples[:MAX_SAMPLES]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False)+'\n')
    long_c = sum(1 for s in samples if '做多' in s['conversations'][-1]['content'])
    short_c = sum(1 for s in samples if '做空' in s['conversations'][-1]['content'])
    print(f"Wrote {len(samples)} samples. Long: {long_c}, Short: {short_c}")

if __name__ == "__main__":
    main()
