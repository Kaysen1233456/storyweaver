"""SFT 对话模板: 数据构建、训练、推理三处共用, 保证格式完全一致。

一条 SFT 样本 = format_prompt(指令) + 故事 + <|endoftext|>
训练时 format_prompt(...) 这部分的 target 置 -100(不计分), 只对故事和结束符计分。
"""

PROMPT_TEMPLATE = 'Instruction: {instruction}\nStory:\n'


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())
