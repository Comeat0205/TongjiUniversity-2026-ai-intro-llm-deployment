from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
model_name = "/mnt/data/Qwen-7B-Chat"

# 你要测试的问题列表
prompts = [
    "请说出以下两句话区别在哪里？1、冬天:能穿多少穿多少 2、夏天:能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思。请问：以上“意思”分别是什么意思。",
    "“下雨天留客天留我不留”，请用不同断句方式解释这句话的含义。"
]

print("正在加载模型...")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 加载模型（CPU兼容版，不依赖accelerate）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

# 禁用梯度计算，加速推理
torch.set_grad_enabled(False)

print("模型加载完成！开始回答问题：\n")

# 循环处理每个问题
for i, prompt in enumerate(prompts, 1):
    print(f"===== 问题 {i} =====")
    print(f"用户：{prompt}")
    print("回答：")

    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    # 流式输出回答
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=False
    )
    print("\n" + "="*50 + "\n")
