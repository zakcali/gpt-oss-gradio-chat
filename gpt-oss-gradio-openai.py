import gradio as gr
from openai import OpenAI
import time  # <-- 1. Import the 'time' module

# Initialize the OpenAI client to connect to a local server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Point to your local server
    api_key="EMPTY"                       # Use a placeholder API key
)

# Main chat function adapted for the OpenAI client
def chat_with_openai(message, history, instructions,
                     temperature, max_tokens, effort):
    
    if not message.strip():
        return history, ""

    # Append user message and a placeholder for the assistant's response
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]

    # Build the list of messages for the API call
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
            max_tokens=int(max_tokens),
            stream=True,
            reasoning_effort=effort,
        )
        
        full_content = ""
        # --- 2. ADD UI THROTTLING LOGIC ---
        last_yield_time = time.time()
        flush_interval_s = 0.04  # Corresponds to ~25 UI updates per second

        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_content += delta.content
                history[-1]["content"] = full_content
                
                # Check if it's time to update the UI
                now = time.time()
                if now - last_yield_time >= flush_interval_s:
                    last_yield_time = now
                    yield history, None

        # Ensure the final, complete response is always sent
        yield history, ""

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        history[-1]["content"] = error_message
        yield history, ""


# --- Gradio UI ---
with gr.Blocks(title="💬 Local LLM Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by Local OpenAI-Compatible API)")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, type="messages")

            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            
            # --- 3. ADD STOP BUTTON AND REARRANGE BUTTONS ---
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### Model: `openai/gpt-oss-120b`")
            
            instructions = gr.Textbox(
                label="System Instructions",
                value="You are a helpful assistant.",
                lines=3
            )
            
            effort = gr.Radio(
                ["low", "medium", "high"], 
                value="medium", 
                label="Reasoning effort"
            )

            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(100, 65535, value=8192, step=256, label="Max Tokens")

    inputs = [msg, chatbot, instructions,
              temperature, max_tokens, effort]
    
    # --- 4. WIRE UP EVENTS FOR CANCELLATION ---
    # Save event handles to be able to cancel them
    e_submit = msg.submit(chat_with_openai, inputs, [chatbot, msg])
    e_click = send_btn.click(chat_with_openai, inputs, [chatbot, msg])

    # Stop button cancels any in-progress generation
    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)

    # Clear button also cancels in-progress generation before clearing
    clear_btn.click(lambda: [], outputs=chatbot, cancels=[e_submit, e_click], queue=False)

# --- 5. ENABLE THE QUEUE FOR SMOOTH STREAMING AND CANCELLATION ---
demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="192.168.0.xx", server_port=7860)
