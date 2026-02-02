import os
import gradio as gr
from openai import OpenAI
import time
import tempfile  #  Import the tempfile module
import atexit    #  Import atexit for cleanup on exit

# This list will hold the paths of all generated chat logs for this session.
temp_files_to_clean = []

# --- Function to perform cleanup on exit ---
def cleanup_temp_files():
    """Iterates through the global list and deletes the tracked files."""
    if not temp_files_to_clean:
        return
    print(f"\nCleaning up {len(temp_files_to_clean)} temporary files...")
    for file_path in temp_files_to_clean:
        try:
            os.remove(file_path)
            # print(f"  - Removed: {file_path}") # Uncomment for verbose logging
        except FileNotFoundError:
            # File might have been moved or deleted by other means
            pass
        except Exception as e:
            # Catch other potential errors (e.g., permissions)
            print(f"  - Error removing {file_path}: {e}")
    print("Cleanup complete.")

# This will be called on normal exit and for most unhandled exceptions,
# including KeyboardInterrupt from Ctrl+C.
atexit.register(cleanup_temp_files)


# The tempfile module will handle creating files in the OS's temporary directory.
print("Temporary chat download files will be saved in the OS's default temp directory.")

# Initialize the OpenAI client to connect to a local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

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

        # --- Use tempfile to create a secure temporary file ---
        # Create a temporary markdown file to store the chat response.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
            output_filepath = temp_file.name
            temp_file.write(full_content)
        
        # Track this file so it can be cleaned up when the app exits
        temp_files_to_clean.append(output_filepath)
        print(f"Created and tracking temp file: {output_filepath}")
        
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
            # chatbot = gr.Chatbot(height=500, type="messages", show_copy_button=True)
            # NEW (Gradio 6.0)
            chatbot = gr.Chatbot(height=500, buttons=["copy"])
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
                download_btn = gr.DownloadButton("⬇️ Download Last Response", visible=False, scale=3)

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
    
demo.queue()

if __name__ == "__main__":
    # --- print messages for clarity ---
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
    demo.launch(server_name="192.168.0.35", server_port=7860)
