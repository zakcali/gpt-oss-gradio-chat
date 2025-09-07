import os
import gradio as gr
from groq import Groq
import time
import tempfile  # --- MODIFICATION: Import the tempfile module
import atexit    # --- MODIFICATION: Import atexit for cleanup on exit

# --- MODIFICATION: Global list to track temporary files ---
# This list will hold the paths of all generated chat logs for this session.
temp_files_to_clean = []

# --- MODIFICATION: Function to perform cleanup on exit ---
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

# --- MODIFICATION: Register the cleanup function to run on script exit ---
# This will be called on normal exit and for most unhandled exceptions,
# including KeyboardInterrupt from Ctrl+C.
atexit.register(cleanup_temp_files)


# --- MODIFICATION: Directory setup is no longer needed ---
# The tempfile module will handle creating files in the OS's temporary directory.
print("Temporary chat download files will be saved in the OS's default temp directory.")
# --- END MODIFICATION ---

# Initialize the Groq client (API key from environment variable)
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# --- MODIFICATION: The manual clear function is no longer needed ---
# --- END MODIFICATION ---

# Main chat function
def chat_with_groq(message, history, model_choice, instructions,
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
        yield history, None, "*Reasoning...*", initial_download_update

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
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            new_content = getattr(delta, "content", None) or None
            new_reasoning = getattr(delta, "reasoning", None) or None
            
            if new_content is not None:
                full_content += new_content
                history[-1]["content"] = full_content

            if new_reasoning is not None:
                reasoning_content += new_reasoning
                
            now = time.time()
            if now - last_yield_time >= flush_interval_s:
                last_yield_time = now
                yield history, None, reasoning_content, initial_download_update

        # --- MODIFICATION: Use tempfile to create a secure temporary file ---
        # Create a temporary markdown file to store the chat response.
        # `delete=False` is crucial so Gradio can access it after this function returns.
        # `mode="w"` and `encoding="utf-8"` ensure correct writing of text.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
            output_filepath = temp_file.name
            temp_file.write(full_content)
        
        # Track this file so it can be cleaned up when the app exits
        temp_files_to_clean.append(output_filepath)
        print(f"Created and tracking temp file: {output_filepath}")
        # --- END MODIFICATION ---
        
        # Create the update object for the download button, pointing to the new path
        final_download_update = gr.update(visible=True, value=output_filepath)
        
        # Final yield to show the completed response and the download button
        yield history, "", reasoning_content, final_download_update

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        yield history, "", f"An error occurred: {e}", initial_download_update


# Gradio UI 
with gr.Blocks(title="💬 Groq Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by Groq)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="messages", show_copy_button=True)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
                # --- MODIFICATION: Removed the manual "Clear Downloads" button ---
                download_btn = gr.DownloadButton("⬇️ Download Last Response", visible=False, scale=3)
                # --- END MODIFICATION ---

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
    
    outputs = [chatbot, msg, thoughts_box, download_btn]

    e_submit = msg.submit(chat_with_groq, inputs, outputs)
    e_click = send_btn.click(chat_with_groq, inputs, outputs)

    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click])

    clear_btn.click(
        lambda: ([], "*Reasoning from gpt-oss-120b will appear here...*", gr.update(visible=False)), 
        outputs=[chatbot, thoughts_box, download_btn], 
        cancels=[e_submit, e_click]
    )

    # --- MODIFICATION: Removed event handler for the deleted button ---
    # --- END MODIFICATION ---

demo.queue()

if __name__ == "__main__":
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
    demo.launch()