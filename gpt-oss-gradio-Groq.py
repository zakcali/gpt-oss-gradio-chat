import os
import gradio as gr
from groq import Groq
import time

# Initialize the Groq client (API key from environment variable)
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Main chat function
def chat_with_groq(message, history, model_choice, instructions,
                   temperature, max_tokens, effort):
    
    if not message.strip():
        return history, "", "*No reasoning generated yet...*"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]

    messages = []
    if instructions.strip():
        messages.append({"role": "system", "content": instructions})

    for m in history:
        if m["role"] == "assistant" and m["content"] == "":
            continue
        messages.append({"role": m["role"], "content": m["content"]})

    try:
        request_params = dict(
            model=model_choice,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        if model_choice == "openai/gpt-oss-120b":
            request_params["reasoning_effort"] = effort
            request_params["top_p"] = 1.0
        elif model_choice == "qwen/qwen3-32b":
            request_params["reasoning_format"] = "hidden"
        elif model_choice == "deepseek-r1-distill-llama-70b":
            request_params["reasoning_format"] = "hidden"

        completion = client.chat.completions.create(**request_params)
        full_content = ""
        reasoning_content = "" 
        last_yield_time = time.time()
        flush_interval_s = 0.04

        for chunk in completion:
            # Check for choices and delta, which might be empty in some chunks
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta

            # --- START OF CORRECTED LOGIC ---

            # 1. Accumulate main content if it exists
            if delta.content:
                full_content += delta.content
                history[-1]["content"] = full_content

            # 2. Accumulate reasoning content if it exists
            if delta.reasoning:
                reasoning_content += delta.reasoning

            # --- END OF CORRECTED LOGIC ---

            now = time.time()
            if now - last_yield_time >= flush_interval_s:
                last_yield_time = now
                yield history, None, reasoning_content

        yield history, "", reasoning_content

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        yield history, "", f"An error occurred: {e}"


# Gradio UI (No changes needed here)
with gr.Blocks(title="💬 Groq Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by Groq)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)

        with gr.Column(scale=1):
            model_choice = gr.Radio(
                ["openai/gpt-oss-120b", "qwen/qwen3-32b", "llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b"], 
                value="openai/gpt-oss-120b", 
                label="Model"
            )
            instructions = gr.Textbox(label="System Instructions", value="You are a helpful assistant.", lines=3)
            effort = gr.Radio(["low", "medium", "high"], value="medium", label="Reasoning effort")
            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(100, 65535, value=8192, step=128, label="Max Tokens")
            
            thoughts_box = gr.Markdown(
                label="🧠 Model Thoughts",
                value="*Reasoning from gpt-oss-120b will appear here...*",
            )

    inputs = [msg, chatbot, model_choice, instructions,
              temperature, max_tokens, effort]
    
    outputs = [chatbot, msg, thoughts_box]

    e_submit = msg.submit(chat_with_groq, inputs, outputs)
    e_click = send_btn.click(chat_with_groq, inputs, outputs)

    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click])

    clear_btn.click(
        lambda: ([], "*Reasoning from gpt-oss-120b will appear here...*"), 
        outputs=[chatbot, thoughts_box], 
        cancels=[e_submit, e_click]
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()