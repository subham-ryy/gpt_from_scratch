"""Formats dialogue and QA datasets into User: / Assistant: conversational turns."""
def format_chat_prompt(messages: list[dict]) -> str:
    """Format dialogue list into a prompt."""
    prompt = ""
    for msg in messages:
        role = msg.get("role", "User").capitalize()
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return prompt

