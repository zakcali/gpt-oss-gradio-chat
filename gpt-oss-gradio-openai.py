import gradio as gr
from openai import OpenAI

# 1. Initialize the OpenAI client to connect to a local server
#    This replaces the Groq client initialization.
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Point to your local server
    api_key="EMPTY"                       # Use a placeholder API key
)

# 2. Main chat function is renamed and adapted for the OpenAI client
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
        # Don't include the empty assistant placeholder in the API call
        if m["role"] == "assistant" and m["content"] == "":
            continue
        messages.append({"role": m["role"], "content": m["content"]})

    try:
        # 3. The API call now uses the OpenAI client.
        #    The model is hardcoded, and custom params are simplified.
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b", # Specify the model served locally
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens), # Ensure max_tokens is an integer
            stream=True,
            reasoning_effort=effort, # Pass the reasoning_effort parameter
        )
        
        full_content = ""
        # The streaming logic is the same, as both libraries follow the same pattern
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_content += delta.content
                history[-1]["content"] = full_content
                # Yield updates to the Gradio UI
                yield history, ""

        # Final update to ensure the full message is set
        history[-1]["content"] = full_content
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

            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            # 4. UI Simplified: Removed the model selector as we target a specific local model
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

    # 5. The list of inputs for the function is updated (model_choice removed)
    inputs = [msg, chatbot, instructions,
              temperature, max_tokens, effort]
    
    # 6. Event handlers now call the new 'chat_with_openai' function
    msg.submit(chat_with_openai, inputs, [chatbot, msg])
    send_btn.click(chat_with_openai, inputs, [chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(server_name="192.168.0.xx", server_port=7860)