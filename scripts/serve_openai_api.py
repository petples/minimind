import argparse
import json
import re
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import traceback
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()
API_TOKEN = "TQaWTatdp5MD782Apm-LIgSNmbAUWZyb-5ptjwS4UqM"  # MiniMind API 認證 token


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location='cpu'), strict=True)
        # 抑制特殊 token 生成
        model._suppress_ids = [0, 1, 2, 25, 26]  # pad, im_start, im_end, <think>, </think>
        if args.lora_weight != 'None':
            apply_lora(model)
            lora_names = [n.strip() for n in args.lora_weight.split(',')]
            lora_dir = f'../{args.save_dir}/lora'
            if len(lora_names) == 1:
                load_lora(model, f'{lora_dir}/{lora_names[0]}_{args.hidden_size}.pth')
            else:
                # Merge multiple LoRAs: sum state_dicts (additive stacking)
                merged = None
                for name in lora_names:
                    path = f'{lora_dir}/{name}_{args.hidden_size}.pth'
                    w = torch.load(path, map_location='cpu')
                    if merged is None:
                        merged = w
                    else:
                        for k, v in w.items():
                            merged[k] = merged[k] + v
                # Load merged weights into LoRA params
                model_state = model.state_dict()
                for k, v in merged.items():
                    if k in model_state:
                        model_state[k].copy_(v.to(model_state[k].device))
                stacked = ' + '.join(lora_names)
                print(f'[LoRA] Stacked {len(lora_names)} adapters: {stacked}')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.half().eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.85
    max_tokens: int = 512
    stream: bool = True
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: dict = None
    
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        safe_max_len = max(1, args.max_seq_len - max_tokens)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=safe_max_len).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.20,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        while True:
            text = queue.get()
            if text is None:
                break
            full_text += text

            if not thinking_ended:
                pos = full_text.find('</think>')
                if pos >= 0:
                    thinking_ended = True
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    emitted = pos + len('</think>')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        traceback.print_exc()
        yield json.dumps({"error": {"message": str(e), "type": "server_error"}}, ensure_ascii=False)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, authorization: str | None = Header(None)):
    """OpenAI-compatible chat completions endpoint."""
    # ── Token 验证 ──
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format, use: Bearer <token>")
    if authorization[7:] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API token")
    # ── 參數驗證 ──
    if request.max_tokens < 1 or request.max_tokens > args.max_seq_len:
        raise HTTPException(status_code=400, detail=f"max_tokens must be between 1 and {args.max_seq_len}")
    if request.temperature < 0 or request.temperature > 2:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
    if request.top_p < 0 or request.top_p > 1:
        raise HTTPException(status_code=400, detail="top_p must be between 0 and 1")
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            safe_max_len = max(1, args.max_seq_len - request.max_tokens)
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=safe_max_len).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=request.max_tokens,
                    do_sample=True,
                    repetition_penalty=1.20,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"service": "MiniMind OpenAI API", "model": args.weight, "version": "v1.0", "endpoints": {"/v1/chat/completions": "POST", "/health": "GET"}}


@app.get("/health")
async def health():
    return {"status": "ok", "model": args.weight, "device": str(device)}


@app.get("/debug/suppress")
async def debug_suppress():
    ids = getattr(model, '_suppress_ids', 'NOT SET')
    return {"_suppress_ids": ids, "model_type": type(model).__name__}


@app.post("/debug/generate")
async def debug_generate(request: ChatRequest):
    """Debug endpoint: run generation and return raw token IDs."""
    new_prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len - request.max_tokens).to(device)
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=request.max_tokens, do_sample=True, temperature=request.temperature, top_p=request.top_p, eos_token_id=2)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return {
        "num_tokens": len(new_tokens),
        "token_ids": new_tokens.tolist(),
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "text_raw": tokenizer.decode(new_tokens, skip_special_tokens=False)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称，多个用逗号分隔（如 lora_elementary,lora_middle_school），None=不使用")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--api_token', default=None, type=str, help="API 认证 token（Bearer），不设置则跳过认证")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device
    if args.api_token:
        os.environ["MINIMIND_API_TOKEN"] = args.api_token
    model, tokenizer = init_model(args)
    if args.api_token:
        print(f"🔐 API token 认证已启用: {args.api_token[:8]}...")
    else:
        print("⚠️  未设置 API token，允许无认证访问")
    uvicorn.run(app, host="0.0.0.0", port=8999)
