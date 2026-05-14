"""
Build Brain Trading SFT dataset from PPO backtest case files.
Generates MiniMind-format conversations with trading signal analysis.
"""

import json
import random
import os
import numpy as np

random.seed(42)
np.random.seed(42)

# ─── Configuration ───
INPUT_PATH = "/root/brain/data/ppo_cases_US_SPY.json"
OUTPUT_PATH = "/root/minimind/dataset/brain_trading_sft.jsonl"
MAX_SAMPLES = 3000

# ─── System prompt ───
SYSTEM_PROMPT = "你是一个专业的量化交易分析师。根据提供的市场数据，判断应该做多还是做空，并给出简要理由。"

# ─── Feature name mapping (for the varying dimensions we have) ───
# Based on analysis of the 14-dim feature vector from the PPO cases:
#   dim 0: price momentum / relative position  [-1.09, 0.83]
#   dim 1: order flow signal                    [-0.44, 0.92]
#   dim 2: VWAP / micro-price deviation         [-0.26, 0.75]
#   dim 4: volatility proxy (volume-adj)        [3.4, 11.6]
#   dim 11: spread / liquidity pressure         [-0.28, 1.25]
#   (other dims are constant / zero)
# We'll map these to named trading features and add synthetic RSI, VPIN, etc.

FEATURE_TEMPLATES = {
    "price_momentum": 0,        # dim 0 → maps to "价格动能"
    "order_flow": 1,            # dim 1 → maps to "订单流"
    "micro_price_dev": 2,       # dim 2 → maps to "微观价格偏离"
    "volatility_proxy": 4,      # dim 4 → maps to "波动率代理"
    "liquidity_pressure": 11,   # dim 11 → maps to "流动性压力"
}

# ─── Load real case data ───
def load_cases():
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
    
    cases = []
    for c in data:
        feat_str = c['state_features']['features']
        feat_vals = [float(x) for x in feat_str.replace('[','').replace(']','').replace('\n',' ').split() if x]
        
        cases.append({
            "features_raw": feat_vals,
            "action_side": c['action']['side'],
            "action_size": c['action']['size'],
            "reward": c['outcome']['reward'],
            "step": c['outcome']['step'],
            "drawdown": c['outcome']['drawdown'],
            "total_cost": c['outcome']['total_cost'],
            "llm_annotation": c.get('llm_annotation'),
        })
    return cases

def derive_named_features(feat_vals):
    """Convert raw feature vector to named trading features with realistic scaling."""
    price_momentum = feat_vals[0]  # -1.1 to 0.83
    order_flow = feat_vals[1]      # -0.44 to 0.92
    micro_dev = feat_vals[2]       # -0.26 to 0.75
    vol_proxy = feat_vals[4]       # 3.4 to 11.6
    liq_pressure = feat_vals[11]   # -0.28 to 1.25
    
    # Derive realistic feature values
    price = 4500.0 + price_momentum * 200.0  # ~4280-4670
    price = round(price, 2)
    
    # RSI: map momentum to 30-70 range
    rsi = 50.0 + price_momentum * 35.0
    rsi = round(np.clip(rsi, 25, 75), 1)
    
    # VPIN: map order flow + liquidity pressure
    vpin = 0.35 + (order_flow * 0.3) + (liq_pressure * 0.15)
    vpin = round(np.clip(vpin, 0.15, 0.85), 2)
    
    # Volatility (20-day)
    vol_20 = 0.10 + (vol_proxy - 3.4) / (11.6 - 3.4) * 0.25
    vol_20 = round(np.clip(vol_20, 0.08, 0.40), 3)
    
    # Turtle channel position
    turtle = 0.3 + price_momentum * 0.4 + order_flow * 0.2
    turtle = round(np.clip(turtle, 0.05, 0.95), 2)
    
    # Order book imbalance
    ob_imbalance = order_flow * 0.8
    ob_imbalance = round(np.clip(ob_imbalance, -0.70, 0.70), 2)
    
    # Fractional diff
    frac_diff = price_momentum * 0.04 + np.random.normal(0, 0.005)
    frac_diff = round(np.clip(frac_diff, -0.08, 0.08), 4)
    
    # Clenow quality
    clenow = 0.5 + price_momentum * 0.3 + (vol_proxy - 5) / 10.0 * 0.2
    clenow = round(np.clip(clenow, 0.30, 0.95), 2)
    
    # Sakata intensity
    sakata = 0.5 + order_flow * 0.35 + liq_pressure * 0.15
    sakata = round(np.clip(sakata, 0.20, 0.95), 2)

    return {
        "price": price,
        "rsi": rsi,
        "vpin": vpin,
        "volatility_20": vol_20,
        "turtle_channel_position": turtle,
        "order_book_imbalance": ob_imbalance,
        "fractional_diff": frac_diff,
        "clenow_quality": clenow,
        "sakata_intensity": sakata,
    }

# ─── Build user prompt from features ───
def build_user_prompt(feats):
    """Build a user prompt string from named features."""
    parts = [
        f"SPX 当前价格 {feats['price']}",
        f"RSI {feats['rsi']}",
        f"VPIN {feats['vpin']}",
        f"波动率(20日) {feats['volatility_20']}",
        f"分數差分(d=0.4) {feats['fractional_diff']}",
        f"海龟通道位置 {feats['turtle_channel_position']}",
        f"订单簿不平衡 {feats['order_book_imbalance']}",
        f"Clenow质量 {feats['clenow_quality']}",
        f"Sakata强度 {feats['sakata_intensity']}",
    ]
    return "，".join(parts)

# ─── Generate assistant response ───
def build_assistant_response(side, feats, label=None, annotation=None):
    """
    Build assistant response. 
    `side`: "long" → 做多, "short" → 做空
    Features are used to construct a short rationale.
    """
    direction = "做多" if side == "long" else "做空"
    
    reasons = []
    
    # VPIN reasoning
    vpin = feats['vpin']
    if vpin > 0.55:
        reasons.append(f"VPIN {vpin} 偏高，存在逆向选择风险")
    elif vpin < 0.30:
        reasons.append(f"VPIN {vpin} 较低，信息不对称风险可控")
    else:
        reasons.append(f"VPIN {vpin} 处于中等水平")
    
    # RSI reasoning
    rsi = feats['rsi']
    if rsi > 65:
        reasons.append(f"RSI {rsi} 处于超买区域")
    elif rsi < 35:
        reasons.append(f"RSI {rsi} 处于超卖区域")
    elif rsi > 55:
        reasons.append(f"RSI {rsi} 偏多")
    elif rsi < 45:
        reasons.append(f"RSI {rsi} 偏空")
    else:
        reasons.append(f"RSI {rsi} 中性")
    
    # Turtle channel reasoning
    turtle = feats['turtle_channel_position']
    if turtle > 0.80:
        reasons.append(f"海龟通道位置 {turtle} 接近上轨，存在回调压力")
    elif turtle < 0.20:
        reasons.append(f"海龟通道位置 {turtle} 接近下轨，存在反弹机会")
    
    # Order book reasoning
    ob = feats['order_book_imbalance']
    if abs(ob) > 0.3:
        direction_word = "买方" if ob > 0 else "卖方"
        reasons.append(f"订单簿不平衡 {ob} 显示{direction_word}压力")
    
    # Clenow quality
    cq = feats['clenow_quality']
    if cq > 0.75:
        reasons.append(f"Clenow质量 {cq}，趋势质量良好")
    elif cq < 0.45:
        reasons.append(f"Clenow质量 {cq} 偏低，趋势可靠性不足")
    
    # Sakata intensity
    si = feats['sakata_intensity']
    if si > 0.75:
        reasons.append(f"Sakata强度 {si}，市场动量充足")
    elif si < 0.40:
        reasons.append(f"Sakata强度 {si} 偏弱，动量不足")
    
    # Volatility
    vol = feats['volatility_20']
    if vol > 0.25:
        reasons.append(f"波动率 {vol} 较高，需注意仓位管理")
    elif vol < 0.12:
        reasons.append(f"波动率 {vol} 偏低，市场较为平稳")
    
    # Use annotation if available (it's null in our data, but keep for extensibility)
    if annotation:
        if annotation.get('success_conditions') and side == 'long':
            reasons.append(f"成功条件: {annotation['success_conditions']}")
        if annotation.get('failure_reason'):
            reasons.append(f"注意: {annotation['failure_reason']}")
    
    # Select 2-3 reasons
    selected = random.sample(reasons, min(len(reasons), random.randint(2, 4)))
    
    if random.random() < 0.3:
        # Simpler format
        response = f"建议{direction}。{'；'.join(selected[:2])}。"
    else:
        response = f"建议{direction}。{'；'.join(selected[:3])}。"
    
    return response

# ─── Main dataset generation ───
def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} cases from {INPUT_PATH}")
    
    # Separate wins (label=1) and neutral (label=0)
    wins = [c for c in cases if c['reward'] > 0]
    neutral = [c for c in cases if c['reward'] == 0]
    print(f"Wins: {len(wins)}, Neutral: {len(neutral)}")
    
    samples = []
    
    # ─── Group A: Real-data-anchored samples ───
    # Use actual feature vectors from cases
    # All real cases are "long" - we'll use them as-is for long signals
    # and also create synthetic short variants
    
    # From wins (200): use all of them
    for case in wins:
        feats = derive_named_features(case['features_raw'])
        user = build_user_prompt(feats)
        assistant = build_assistant_response('long', feats)
        samples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    
    # From neutral (2800): use a diverse subset
    random.shuffle(neutral)
    # Take about 800 neutral longs
    neutral_longs = neutral[:800]
    for case in neutral_longs:
        feats = derive_named_features(case['features_raw'])
        user = build_user_prompt(feats)
        assistant = build_assistant_response('long', feats)
        samples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    
    print(f"Long samples (from real cases): {len(samples)}")
    
    # ─── Group B: Synthetic short samples ───
    # Create short variants by flipping some features and using short rationale
    target_total = min(MAX_SAMPLES, 3000)
    short_needed = target_total - len(samples)
    short_needed = max(short_needed, 1000)  # at least 1000 shorts
    
    # Generate shorts using feature vectors from all cases (with perturbations)
    all_cases_pool = cases.copy()
    random.shuffle(all_cases_pool)
    
    for i in range(short_needed):
        base_case = all_cases_pool[i % len(all_cases_pool)]
        # Perturb features to simulate short conditions
        perturbed = base_case['features_raw'].copy()
        # Flip momentum-related features for short signal
        perturbed[0] = np.clip(-perturbed[0] + random.uniform(-0.2, 0.2), -1.1, 0.9)
        perturbed[1] = np.clip(-perturbed[1] + random.uniform(-0.15, 0.15), -0.9, 0.5)
        perturbed[2] = np.clip(-perturbed[2] + random.uniform(-0.1, 0.1), -0.8, 0.3)
        perturbed[11] = np.clip(-perturbed[11] + random.uniform(-0.1, 0.3), -1.3, 0.3)
        
        feats = derive_named_features(perturbed)
        user = build_user_prompt(feats)
        assistant = build_assistant_response('short', feats)
        samples.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    
    print(f"Short samples (synthetic): {short_needed}")
    
    # ─── Group C: Additional synthetic long samples with varied features ───
    remaining = target_total - len(samples)
    if remaining > 0:
        for i in range(remaining):
            base_case = random.choice(all_cases_pool)
            perturbed = base_case['features_raw'].copy()
            # Add random noise
            perturbed[0] = np.clip(perturbed[0] + random.uniform(-0.3, 0.3), -1.1, 0.9)
            perturbed[1] = np.clip(perturbed[1] + random.uniform(-0.2, 0.2), -0.9, 0.9)
            perturbed[2] = np.clip(perturbed[2] + random.uniform(-0.15, 0.15), -0.8, 0.8)
            perturbed[11] = np.clip(perturbed[11] + random.uniform(-0.2, 0.2), -1.3, 1.3)
            
            feats = derive_named_features(perturbed)
            user = build_user_prompt(feats)
            assistant = build_assistant_response('long', feats)
            samples.append({
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            })
    
    # Shuffle and write
    random.shuffle(samples)
    samples = samples[:MAX_SAMPLES]
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")
    
    # ─── Quick stats ───
    long_count = sum(1 for s in samples if '做多' in s['conversations'][-1]['content'])
    short_count = sum(1 for s in samples if '做空' in s['conversations'][-1]['content'])
    print(f"  Long signals: {long_count}")
    print(f"  Short signals: {short_count}")
    
    # Print a few examples
    print("\n─── Sample conversations ───")
    for i, s in enumerate(random.sample(samples, 3)):
        print(f"\n--- Sample {i+1} ---")
        for msg in s['conversations']:
            print(f"[{msg['role']}]: {msg['content'][:200]}...")
        print()

if __name__ == "__main__":
    main()
