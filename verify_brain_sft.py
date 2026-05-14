"""Verify Brain SFT model on trading questions."""
import torch
import sys
sys.path.insert(0, '/root/minimind')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

config = MiniMindConfig()
model = MiniMindForCausalLM(config)
state = torch.load('/root/minimind/out/brain_sft_768.pth', map_location='cpu')
model.load_state_dict(state, strict=False)
tokenizer = AutoTokenizer.from_pretrained('/root/minimind/model')
model.eval()

test_prompts = [
    "系统：你是一个专业的量化交易分析师。根据提供的市场数据，判断应该做多还是做空。\n用户：SPX 当前价格 4580.2，RSI 55，VPIN 0.45，波动率(20日) 0.18，分數差分(d=0.4) 0.023，海龟通道位置 0.92，订单簿不平衡 0.12，Clenow质量 0.73，Sakata强度 0.88\n助手：",
    "系统：你是一个专业的量化交易分析师。根据提供的市场数据，判断应该做多还是做空。\n用户：SPX 当前价格 4320.5，RSI 28，VPIN 0.21，波动率(20日) 0.35，分數差分(d=0.4) -0.045，海龟通道位置 0.08，订单簿不平衡 -0.55，Clenow质量 0.31，Sakata强度 0.22\n助手：",
    "系统：你是一个专业的量化交易分析师。根据提供的市场数据，判断应该做多还是做空。\n用户：SPX 当前价格 4650.0，RSI 72，VPIN 0.68，波动率(20日) 0.12，分數差分(d=0.4) 0.058，海龟通道位置 0.88，订单簿不平衡 0.45，Clenow质量 0.82，Sakata强度 0.91\n助手：",
]

for i, prompt in enumerate(test_prompts):
    print(f"\n{'='*60}")
    print(f"Test {i+1}:")
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Show just the assistant part
    if '助手：' in response:
        assistant_part = response.split('助手：')[-1]
        print(f"Generated: {assistant_part[:300]}")
    else:
        print(f"Full: {response[:400]}")
