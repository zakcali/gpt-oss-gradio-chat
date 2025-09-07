import os
import gradio as gr
from openai import OpenAI
import time

# Create a dedicated directory for temporary chat downloads
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(APP_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
print(f"Temporary chat logs will be saved in: {TEMP_DIR}")

# Initialize the OpenAI client to connect to a local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def clear_temp_folder():
    """Deletes all files in the TEMP_DIR and returns a status message."""
    count = 0
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            # Make sure it's a file before trying to delete
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        
        if count > 0:
            return f"✅ Cleared {count} file(s) from the temp downloads directory."
        else:
            return "ℹ️ Temp downloads directory is already empty."
            
    except Exception as e:
        print(f"Error clearing temp folder: {e}")
        return f"❌ Error clearing temp folder: {e}"


def chat_with_openai(message, history, instructions,
                     temperature, max_tokens, effort):
    
    initial_download_update = gr.update(visible=False)

    if not message.strip():
        return history, "", "*No reasoning generated yet...*", initial_download_update

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
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            max_tokens=int(max_tokens),
            stream=True,
            reasoning_effort=effort,
        )
        
        full_content = ""
        reasoning_content = ""
        last_yield_time = time.time()
        flush_interval_s = 0.04

        for chunk in completion:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            new_content = getattr(delta, "content", None) or None
            # Note: The local server might use 'reasoning' or 'reasoning_content'
            new_reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)

            if new_content is not None:
                full_content += new_content
                history[-1]["content"] = full_content

            if new_reasoning is not None:
                reasoning_content += new_reasoning
                
            now = time.time()
            if now - last_yield_time >= flush_interval_s:
                last_yield_time = now
                yield history, None, reasoning_content, initial_download_update

        timestamp = int(time.time())
        base_filename = f"chat_response_{timestamp}.md"
        output_filepath = os.path.join(TEMP_DIR, base_filename)
        
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(full_content)
        
        final_download_update = gr.update(visible=True, value=output_filepath)
        
        yield history, "", reasoning_content, final_download_update

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        history[-1]["content"] = error_message
        yield history, "", f"An error occurred: {e}", initial_download_update


# --- Gradio UI  ---
with gr.Blocks(title="💬 Local LLM Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by Local OpenAI-Compatible API)")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="messages", show_copy_button=True)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
                clear_temp_btn = gr.Button("🧹 Clear Downloads", scale=1)
                download_btn = gr.DownloadButton("⬇️ Download Last Response", visible=False, scale=2)

        with gr.Column(scale=1):
            gr.Markdown("### Model: `openai/gpt-oss-120b`")
            instructions = gr.Textbox(label="System Instructions", value="You are a helpful assistant.", lines=3)
            effort = gr.Radio(["low", "medium", "high"], value="medium", label="Reasoning effort")
            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(100, 65535, value=8192, step=256, label="Max Tokens")
            thoughts_box = gr.Markdown(label="🧠 Model Thoughts", value="*Reasoning will appear here...*")

    inputs = [msg, chatbot, instructions, temperature, max_tokens, effort]
    outputs = [chatbot, msg, thoughts_box, download_btn]

    e_submit = msg.submit(chat_with_openai, inputs, outputs)
    e_click = send_btn.click(chat_with_openai, inputs, outputs)
    
    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)
    
    clear_btn.click(
        lambda: ([], "*Reasoning will appear here...*", gr.update(visible=False)), 
        outputs=[chatbot, thoughts_box, download_btn], 
        cancels=[e_submit, e_click], 
        queue=False
    )
    
    clear_temp_btn.click(
        fn=lambda: (clear_temp_folder(), gr.update(visible=False)),
        inputs=None,
        outputs=[thoughts_box, download_btn],
        queue=False
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="192.168.0.35", server_port=7860)