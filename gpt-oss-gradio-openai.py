import gradio as gr
from openai import OpenAI
import time

# Initialize the OpenAI client to connect to a local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def chat_with_openai(message, history, instructions,
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
        # Note: The 'reasoning_effort' parameter might be passed as an 'extra_body'
        # parameter if the library doesn't officially support it. However, if it works
        # as a top-level argument for your server, you can leave it.
        # For max compatibility, it would look like this:
        # extra_body={"reasoning_effort": effort}
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            max_tokens=int(max_tokens),
            stream=True,
            reasoning_effort=effort, # Assuming your local server accepts this
        )
        
        full_content = ""
        reasoning_content = ""
        last_yield_time = time.time()
        flush_interval_s = 0.04

        for chunk in completion:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            # --- START OF MODIFIED EXTRACTION LOGIC ---
            
            # Use getattr() for a safe and concise way to get chunk content
            new_content = getattr(delta, "content", None) or None
            new_reasoning = getattr(delta, "reasoning_content", None) or None

            # Accumulate content if any was extracted
            if new_content is not None:
                full_content += new_content
                history[-1]["content"] = full_content

            # Accumulate reasoning if any was extracted
            if new_reasoning is not None:
                reasoning_content += new_reasoning
                
            # --- END OF MODIFIED EXTRACTION LOGIC ---

            now = time.time()
            if now - last_yield_time >= flush_interval_s:
                last_yield_time = now
                yield history, None, reasoning_content

        yield history, "", reasoning_content

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        history[-1]["content"] = error_message
        yield history, "", f"An error occurred: {e}"


# --- Gradio UI (No changes needed here) ---
with gr.Blocks(title="💬 Local LLM Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by Local OpenAI-Compatible API)")
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
            gr.Markdown("### Model: `openai/gpt-oss-120b`")
            instructions = gr.Textbox(label="System Instructions", value="You are a helpful assistant.", lines=3)
            effort = gr.Radio(["low", "medium", "high"], value="medium", label="Reasoning effort")
            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(100, 65535, value=8192, step=256, label="Max Tokens")
            thoughts_box = gr.Markdown(label="🧠 Model Thoughts", value="*Reasoning will appear here...*")

    inputs = [msg, chatbot, instructions, temperature, max_tokens, effort]
    outputs = [chatbot, msg, thoughts_box]
    e_submit = msg.submit(chat_with_openai, inputs, outputs)
    e_click = send_btn.click(chat_with_openai, inputs, outputs)
    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)
    clear_btn.click(lambda: ([], "*Reasoning will appear here...*"), outputs=[chatbot, thoughts_box], cancels=[e_submit, e_click], queue=False)

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="192.168.0.xx", server_port=7860)