from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# 模型路径
model_path = "/mnt/data/chatglm3-6b"

# 问题列表
questions = [
    "请说出以下两句话区别在哪里？1、冬天:能穿多少穿多少 2、夏天:能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？小明：没什么意思。意思意思。领导：你这就不够意思了。小明：小意思。请问：以上“意思”分别是什么意思。",
    "“下雨天留客天留我不留”，请用不同断句方式解释这句话的含义。"
]

# 加载模型
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

# 禁用梯度，加速推理
torch.set_grad_enabled(False)
print("模型加载完成！开始回答：\n")

# 循环回答所有问题
for i, q in enumerate(questions):
    print(f"===== 问题 {i+1} =====")
    print(f"用户：{q}")

    inputs = tokenizer(q, return_tensors="pt")
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            use_cache=True
        )

    print("\n")
