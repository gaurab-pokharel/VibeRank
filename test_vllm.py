from vllm import LLM, SamplingParams

prompt = """You must answer with exactly one word:
READY

Do not say anything else.
"""

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=False,
    gpu_memory_utilization=0.8,
)

params = SamplingParams(
    temperature=0.0,
    max_tokens=8,
)

outputs = llm.generate([prompt], params)

print("RAW OUTPUT:")
print(repr(outputs[0].outputs[0].text.strip()))

del llm