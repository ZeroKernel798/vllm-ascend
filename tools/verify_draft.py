"""Verify: chain4's top-1 draft == branch2's top-1 draft.
Same speculator (Eagle3), same prompt, max_tokens=1, temperature=0.
If root draft is identical, first output token MUST match.

Model paths follow the same MODEL_PREFIX convention as the e2e tests:
  MODEL_PREFIX=/data/modelscope_cache/models python tools/verify_draft.py
"""
import os

from vllm import LLM, SamplingParams

_PREFIX = os.environ.get("MODEL_PREFIX", "")


def _model(name: str) -> str:
    return f"{_PREFIX}/{name.replace('/', '--')}/snapshots/master" if _PREFIX else name


M8 = _model("Qwen/Qwen3-8B")
E3 = _model("RedHatAI/Qwen3-8B-speculator.eagle3")
PROMPT = 'Explain quantum computing in one sentence:'

spec_configs = {
    'chain4': {
        'method': 'eagle3', 'model': E3,
        'num_speculative_tokens': 4,
        'speculative_token_tree': '[(0,),(0,0),(0,0,0),(0,0,0,0)]'
    },
    'branch2': {
        'method': 'eagle3', 'model': E3,
        'num_speculative_tokens': 3,
        'speculative_token_tree': '[(0,),(0,0),(0,1)]'
    }
}

results = {}
for name, spec in spec_configs.items():
    print(f'Testing {name}...', flush=True)
    llm = LLM(
        model=M8, max_num_seqs=256, gpu_memory_utilization=0.65,
        enforce_eager=True, speculative_config=spec,
        max_model_len=2048, trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=1)
    outputs = llm.generate([PROMPT], sp)
    token_id = outputs[0].outputs[0].token_ids[0]
    token_text = outputs[0].outputs[0].text
    results[name] = {'token_id': token_id, 'token_text': token_text}
    print(f'  first_token: id={token_id}, text={repr(token_text)}', flush=True)
    del llm

print(f'\n=== VERIFICATION ===')
if results['chain4']['token_id'] == results['branch2']['token_id']:
    print(f'PASS: chain4 top-1 == branch2 top-1')
    print(f'  id={results["chain4"]["token_id"]}, text={repr(results["chain4"]["token_text"])}')
    print(f'  Branch mask does NOT corrupt the linear path.')
else:
    print(f'FAIL: chain4={results["chain4"]["token_id"]} != branch2={results["branch2"]["token_id"]}')
