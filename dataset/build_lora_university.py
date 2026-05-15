#!/usr/bin/env python3
"""Generate university-level LoRA training data from annotated PPO cases.

Each annotated case contains: pattern_name, regime_tags, market_phase, confidence.
We generate ShareGPT-format conversations covering:
  - Market analysis & regime identification
  - Risk management & position sizing
  - Strategy reasoning & pattern recognition
  - Loss attribution & lesson learning
"""

import json
import random
from pathlib import Path

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一位專業的量化交易分析師，專精於市場結構分析、風險管理和策略推理。"
    "根據提供的市場數據和案例上下文，給出嚴謹、有理有據的分析判斷，並附上簡明理由。"
)

INPUT_FILE = "/root/brain/data/ppo_cases_annotated.json"
OUTPUT_FILE = "/root/minimind/dataset/lora_university.jsonl"

def parse_features(feature_str):
    """Parse the features string into a list of floats."""
    try:
        return json.loads(feature_str.replace('\n', ' ').replace('[ ', '[').replace(' ]', ']'))
    except Exception:
        return None

def build_qa_pairs(case):
    """Build multiple Q&A conversation pairs from one annotated case."""
    pairs = []

    pat = case["llm_annotation"].get("pattern_name", "unknown")
    regimes = case["llm_annotation"].get("regime_tags", [])
    phase = case["llm_annotation"].get("market_phase", "unknown")
    conf = case["llm_annotation"].get("confidence", 0.5)
    symbol = case.get("symbol", "SPX")
    timeframe = case.get("timeframe", "60m")
    action = case.get("action", {})
    outcome = case.get("outcome", {})
    state = case.get("state_features", {})

    side = action.get("side", "unknown")
    size = action.get("size", 0)
    label = outcome.get("label", 0)
    reward = outcome.get("reward", 0)

    features = state.get("features", [])
    if isinstance(features, str):
        features = parse_features(features) or []
    if not isinstance(features, list):
        features = []

    # ── 1. Market Analysis ──────────────────────────────────────────────────
    if features:
        feat_str = ", ".join([f"{v:.3f}" for v in features[:6]])
        pairs.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"【市場分析】{symbol} ({timeframe}) 市場階段：{phase}，技術特徵：[{feat_str}]，信心度：{conf:.2f}。請分析當前市場結構並給出判斷。"},
                {"role": "assistant", "content": f"市場處於【{phase}】體制。特徵向量分析顯示：{'動量主導' if 'momentum' in regimes else '均值回歸主導' if 'mean_reversion' in regimes else '區間震盪'}格局。{'高' if 'high_volatility' in regimes else '低' if 'low_volatility' in regimes else '中等'}波動率環境，信心度 {conf:.2f}。建議{'順勢操作' if 'momentum' in regimes else '逆勢操作' if 'mean_reversion' in regimes else '區間邊界操作'}。"}
            ]
        })

    # ── 2. Regime Identification ───────────────────────────────────────────
    regime_str = ", ".join(regimes) if regimes else "未識別"
    pairs.append({
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【體制識別】{symbol} 案例，regime_tags：{regime_str}，pattern：{pat}。這屬於哪類市場體制？如何根據這些標籤調整策略？"},
            {"role": "assistant", "content": f"該案例屬於多重體制疊加：{regime_str}。核心 pattern：【{pat}】。策略含義：{'順勢突破' if 'breakout' in regimes else '區間操作' if 'ranging' in regimes else '風險規避' if 'risk_off' in regimes else '趨勢跟隨'}。波動率 regime 決定倉位上限，動量 regime 決定進場方向。"}
        ]
    })

    # ── 3. Risk Management ─────────────────────────────────────────────────
    pairs.append({
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【風險管理】{symbol} 操作方向：{side}，倉位：{size:.2f}，市場階段：{phase}，體制：{regime_str}。如何確定止盈止損位？建議倉位調整？"},
            {"role": "assistant", "content": f"根據 {phase} + {side} 操作情境：{'降低倉位' if 'high_volatility' in regimes or conf < 0.6 else '標準倉位'}（建議系數 {0.5 if 'high_volatility' in regimes else 1.0}），止盈参考波動率倍數，止損不超過資金 2%。pattern {pat} 提示：{'突破失敗風險高' if 'breakout' in pat else '動量持續' if 'momentum' in pat else '震盪整理'}。"}
        ]
    })

    # ── 4. Strategy Reasoning ───────────────────────────────────────────────
    pairs.append({
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【策略推理】{symbol} 案例：pattern={pat}，confidence={conf:.2f}，action={side}×{size:.2f}。這個決策背後的推理邏輯是什麼？"},
            {"role": "assistant", "content": f"決策推理：{phase} 市場中，pattern【{pat}】置信度 {conf:.2f}，{'支持' if conf > 0.6 else '不支持'}進場。{side} 方向符合{'動量方向' if 'momentum' in regimes else '均值回歸方向' if 'mean_reversion' in regimes else '區間邊界反轉'}。倉位 {size:.2f} 體現了{'謹慎' if size < 0.5 else '積極'}的風險態度。"}
        ]
    })

    # ── 5. Pattern Recognition ─────────────────────────────────────────────
    pairs.append({
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【圖形識別】{symbol} 出現 pattern={pat}，regime={regime_str}。這是什麼市場形態？後市可能如何發展？"},
            {"role": "assistant", "content": f"形態識別：【{pat}】{'動量突破型' if 'breakout' in pat else '動量持續型' if 'continuation' in pat else '均值回歸型' if 'reversion' in pat else '整理型'}。體制 {regime_str} 強化此形態可信度。歷史類似案例通常{'延續趨勢' if 'momentum' in regimes else '回歸均值' if 'mean_reversion' in regimes else '區間震盪'}。"}
        ]
    })

    # ── 6. Loss Attribution / Lesson Learning ──────────────────────────────
    if label in (-1, 0) or reward < 0:
        pairs.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"【虧損歸因】{symbol} 案例：pattern={pat}，regime={regime_str}，confidence={conf:.2f}，result={label}。如果這筆交易虧損，核心原因是什麼？"},
                {"role": "assistant", "content": f"虧損歸因分析：pattern【{pat}】在 {phase} 市場失敗通常因為：1) confidence={conf:.2f} {'過低，放量突破未確認' if conf < 0.6 else '尚可但體制判斷失誤'}；2) {regime_str} regime 切换导致趋势逆转；3) 仓位于{'过大' if size > 0.5 else '适中'}风险暴露。改进建议：提高 confidence 阈值至 0.65+，在 {regime_str} 环境降低仓位系数。"}
            ]
        })

    # ── 7. Success Conditions ──────────────────────────────────────────────
    sc = case["llm_annotation"].get("success_conditions")
    if sc:
        pairs.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"【成功條件】{symbol} pattern={pat}，success_conditions：{sc}。在什麼條件下這筆交易應該獲利？如何提高勝率？"},
                {"role": "assistant", "content": f"成功條件：{sc}。執行關鍵：pattern【{pat}】觸發需滿足 {phase} 市場確認，confidence ≥ 0.65，體制標籤一致。提高勝率：在 {regime_str} 環境中等待回調確認進場，止盈設於前高/前低，止損設於突破失敗位。"}
            ]
        })

    # ── 8. Action Validation ───────────────────────────────────────────────
    action_label = "做多" if side == "long" else "做空" if side == "short" else "觀望"
    result_desc = "盈利" if label == 1 else "虧損" if label == -1 else "持平"
    pairs.append({
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【決策驗證】{symbol} 選擇{action_label}，倉位{size:.2f}，結果：{result_desc}。這個決策合理嗎？有何改進空間？"},
            {"role": "assistant", "content": f"決策評估：{action_label} {size:.2f} 仓位在 {phase} + {pat} 情境下{'合理' if conf > 0.65 and label != -1 else '存在疑問'}。結果 {result_desc}（label={label}）。{'成功關鍵：體制判斷準確，倉位管理得當' if label == 1 else '改進方向：提高 confidence 阈值，或等待更清晰信號再進場'}。"}
        ]
    })

    return pairs

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases from {INPUT_FILE}")

    all_samples = []
    for case in cases:
        pairs = build_qa_pairs(case)
        all_samples.extend(pairs)

    print(f"Generated {len(all_samples)} raw samples")

    # Balance: upsample to reach ~10000
    target = 10000
    if len(all_samples) < target:
        # Randomly sample with replacement to reach target
        random.seed(42)
        extras = random.choices(all_samples, k=target - len(all_samples))
        all_samples.extend(extras)
        print(f"Upsampled to {len(all_samples)} samples")

    random.shuffle(all_samples)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_samples)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()