#!/usr/bin/env python3
"""
Build DPO preference pairs from Brain PPO backtest cases.
Produces /root/minimind/dataset/brain_trading_dpo.jsonl

DPO format:
  {"chosen": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}],
   "rejected": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
"""

import json
import random
import numpy as np

# Feature names in order from FeaturePipeline
FEATURE_NAMES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "fracdiff_close", "sakata_intensity", "clenow_quality", "vpin",
    "price_sma20_ratio", "price_sma50_ratio", "bb_position", "rsi_14",
    "momentum_10", "momentum_20"
]

FEATURE_NAMES_CN = [
    "1日收益率", "5日收益率", "10日收益率", "20日收益率",
    "分数差分", "阪田强度", "Clenow质量", "VPIN",
    "价格/20MA比", "价格/50MA比", "布林带位置", "RSI(14)",
    "10日动量", "20日动量"
]

# Per-feature thresholds for bullish / bearish classification.
# (bullish_thresh, bearish_thresh): val > bullish → bullish; val < bearish → bearish.
# VPIN (idx 7): low → bullish (good liquidity); high → bearish (tight liquidity).
BULL_THRESH = [
    0.5,  0.5,  0.5,  0.5,     # returns
    0.5,  0.5,  0.5,  -0.5,    # fracdiff, sakata, clenow, VPIN
    0.3,  0.3,  0.3,  0.5,     # price ratios, bb, rsi
    0.5,  0.5                   # momentum
]
BEAR_THRESH = [
    -0.5, -0.5, -0.5, -0.5,    # returns
    -0.5, -0.5, -0.5,  0.5,    # fracdiff, sakata, clenow, VPIN
    -0.3, -0.3, -0.3, -0.5,    # price ratios, bb, rsi
    -0.5, -0.5                  # momentum
]

SYSTEM_PROMPT = "你是一个专业的量化交易分析师。根据市场数据，判断应该做多还是做空，并给出理由。"


def parse_features(feat_str):
    clean = feat_str.replace('[', '').replace(']', '').replace('\n', ' ').strip()
    return [float(x) for x in clean.split()]


def is_bullish(i, val):
    """True if feature i at value val is a bullish signal."""
    if i == 7:  # VPIN: low → bullish
        return val < -0.5
    return val > BULL_THRESH[i]


def is_bearish(i, val):
    """True if feature i at value val is a bearish signal."""
    if i == 7:  # VPIN: high → bearish
        return val > 0.5
    return val < BEAR_THRESH[i]


def describe_features(features):
    """Produce Chinese natural-language description of the feature vector."""
    lines = []
    for i, name in enumerate(FEATURE_NAMES_CN):
        val = features[i]
        if is_bullish(i, val):
            lines.append(f"  - {name}: {val:.3f}（看涨信号）")
        elif is_bearish(i, val):
            lines.append(f"  - {name}: {val:.3f}（看跌信号）")
        else:
            lines.append(f"  - {name}: {val:.3f}（中性）")
    return "\n".join(lines)


def build_user_prompt(features, position):
    desc = describe_features(features)
    pos_text = {0: "空仓", 1: "持有多头", -1: "持有空头"}.get(position, f"仓位:{position}")
    return f"""当前市场状态如下：
{desc}
当前持仓状态: {pos_text}

请分析市场数据，给出你的交易建议（做多/做空/观望）并说明理由。"""


def collect_bullish(features):
    """Return list of Chinese descriptions for bullish signals."""
    out = []
    for i in range(len(features)):
        val = features[i]
        name = FEATURE_NAMES_CN[i]
        if i == 7:  # VPIN
            if val < -0.5:
                out.append(f"{name}较低（{val:.2f}），市场流动性充足，利于做多")
        else:
            if val > BULL_THRESH[i]:
                out.append(f"{name}={val:.2f}偏高，显示上升动能")
    return out


def collect_bearish(features):
    """Return list of Chinese descriptions for bearish signals."""
    out = []
    for i in range(len(features)):
        val = features[i]
        name = FEATURE_NAMES_CN[i]
        if i == 7:  # VPIN
            if val > 0.5:
                out.append(f"{name}偏高（{val:.2f}），流动性紧张，下行风险增加")
        else:
            if val < BEAR_THRESH[i]:
                out.append(f"{name}={val:.2f}偏低，显示下跌动能")
    return out


# ── Chosen (correct) responses ──────────────────────────────────────

def build_long_reasoning(features):
    """Correct reasoning to go LONG (for win cases)."""
    bullish = collect_bullish(features)
    if bullish:
        return "建议做多。" + "；".join(bullish[:4]) + "。综合来看，多个指标指向上涨趋势，建议顺势做多。"
    return "建议做多。市场整体处于修复阶段，主要技术指标虽未明显超买，但价格结构显示支撑有效，建议轻仓做多。"


def build_wait_reasoning(features):
    """Correct reasoning to WAIT (for loss cases where long was wrong)."""
    return "建议观望。当前市场各技术指标多处于中性区域，缺乏明确的趋势信号。在方向不明朗时，保持空仓等待更好的入场机会是更理性的选择。"


def build_short_reasoning(features):
    """Correct reasoning to SHORT (for loss cases where long was wrong)."""
    bearish = collect_bearish(features)
    if bearish:
        return "建议做空或观望。" + "；".join(bearish[:3]) + "。综合来看，技术指标偏弱，不建议此时做多。"
    return "建议做空或观望。市场方向不明，多个指标处于中性区域，缺乏明确的做多信号，建议等待更明确的突破。"


# ── Rejected (incorrect) responses ──────────────────────────────────

def build_opposite_reasoning(features):
    """Opposite reasoning (for win cases: argue against long)."""
    bearish = collect_bearish(features)
    if bearish:
        return "建议做空。" + "；".join(bearish[:3]) + "。虽然部分指标尚可，但下跌风险不容忽视，建议做空。"
    return "建议观望。当前市场信号混杂，上涨动能不足，追多风险较大，建议等待更明确的方向。"


def build_weak_long_reasoning(features):
    """Weak/flawed long reasoning (rejected for loss cases)."""
    return "建议做多。市场前期上涨后，趋势可能延续，动量指标尚可，可以考虑继续持有多头仓位。虽然部分信号不明确，但趋势跟随策略仍支持做多。"


def build_overconfident_reasoning(features):
    """Overconfident/flawed reasoning (rejected for loss cases)."""
    return "建议重仓做多。当前市场动量强劲，多个指标显示超买但趋势未完，应积极加仓追涨。历史回测表明强势市场应当大胆做多。"


# ── Main ────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    with open('/root/brain/data/ppo_cases_US_SPY.json') as f:
        all_cases = json.load(f)

    wins = [c for c in all_cases if c['outcome']['reward'] > 0]
    losses = [c for c in all_cases if c['outcome']['reward'] == 0]

    print(f"Total cases: {len(all_cases)}")
    print(f"Wins: {len(wins)}, Losses: {len(losses)}")

    # Target ~400 pairs, balanced
    n_each = min(200, len(wins), len(losses))
    sampled_wins = random.sample(wins, n_each)
    sampled_losses = random.sample(losses, n_each)
    print(f"Sampled {n_each} wins + {n_each} losses = {n_each*2} pairs")

    pairs = []

    # ── WIN PAIRS ──
    for case in sampled_wins:
        feats = parse_features(case['state_features']['features'])
        pos = case['state_features'].get('position', 0)
        prompt = build_user_prompt(feats, pos)

        chosen = build_long_reasoning(feats)
        rejected = build_opposite_reasoning(feats)

        pairs.append({
            "chosen": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ],
            "rejected": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
        })

    # ── LOSS PAIRS ──
    for case in sampled_losses:
        feats = parse_features(case['state_features']['features'])
        pos = case['state_features'].get('position', 0)
        prompt = build_user_prompt(feats, pos)

        # Correct answer: don't go long
        if random.random() < 0.5:
            chosen = build_wait_reasoning(feats)
        else:
            chosen = build_short_reasoning(feats)

        # Wrong answer: go long (which is what was done and lost)
        if random.random() < 0.5:
            rejected = build_weak_long_reasoning(feats)
        else:
            rejected = build_overconfident_reasoning(feats)

        pairs.append({
            "chosen": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ],
            "rejected": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}
            ]
        })

    random.shuffle(pairs)

    out_path = '/root/minimind/dataset/brain_trading_dpo.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"Wrote {len(pairs)} pairs to {out_path}")

    # Print samples
    print("\n=== SAMPLE WIN PAIR ===")
    wp = next(p for p in pairs if "做多" in p['chosen'][-1]['content'] and ("做空" in p['rejected'][-1]['content'] or "观望" in p['rejected'][-1]['content']))
    print(f"CHOSEN: {wp['chosen'][-1]['content'][:250]}")
    print(f"REJECTED: {wp['rejected'][-1]['content'][:250]}")

    print("\n=== SAMPLE LOSS PAIR ===")
    lp = next(p for p in pairs if ("观望" in p['chosen'][-1]['content'] or "做空" in p['chosen'][-1]['content']))
    print(f"CHOSEN: {lp['chosen'][-1]['content'][:250]}")
    print(f"REJECTED: {lp['rejected'][-1]['content'][:250]}")

    # Quick sanity check
    print(f"\n=== SANITY CHECK ===")
    with open(out_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines[:2]):
        d = json.loads(line)
        assert len(d['chosen']) == 3, f"chosen len={len(d['chosen'])}"
        assert len(d['rejected']) == 3, f"rejected len={len(d['rejected'])}"
        assert d['chosen'][0]['content'] == SYSTEM_PROMPT
        assert d['rejected'][0]['content'] == SYSTEM_PROMPT
        assert d['chosen'][1]['content'] == d['rejected'][1]['content'], "User prompts differ!"
    print("Format validation: PASSED")


if __name__ == '__main__':
    main()
