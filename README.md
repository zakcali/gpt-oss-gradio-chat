# gpt-oss-gradio-chat

A Gradio chat interface that uses either the OpenAI or Groq API.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/zakcali/gpt-oss-gradio-chat.git
    cd gpt-oss-gradio-chat
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running the "gpt-oss-gradio-Groq.py" script, you need to set your API keys as environment variables.

### For the Groq script:
```bash
export GROQ_API_KEY
```

## Running the Application

You can launch the application in several ways depending on your needs.

### To run locally on your machine:
Use the following line in the script:
```python
demo.launch()
```

### To serve to other devices on your local network (LAN):
Modify the last line to include your machine's local IP address.
```python
# Replace "192.168.0.xx" with your actual LAN IP address
demo.launch(server_name="192.168.0.xx", server_port=7860)
```

### To share publicly over the internet:
Set the `share` parameter to `True`. Gradio will generate a temporary public URL for you.
```python
demo.launch(share=True)
```
For more details, see the official Gradio guide on [Sharing Your App](https://www.gradio.app/guides/sharing-your-app).

## Script Information

*   `gpt-oss-gradio-Groq.py`: This script uses the Groq API and should run on any machine with an internet connection.

*   `gpt-oss-gradio-openai.py`: This script is designed to run a local model and requires substantial hardware. For example, it has been tested on a Linux machine with 4x RTX 3090 GPUs (96 GB total VRAM) using the following `vllm` command to serve the model:

    ```bash
    vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --async-scheduling
    ```
    Reference: [Hugging Face Discussion](https://huggingface.co/openai/gpt-oss-120b/discussions/122)

This script interfaces with a locally served model using [vllm](https://github.com/vllm-project/vllm), a high-throughput LLM serving library. Ensure you have `vllm` installed and a compatible hardware setup before running the server command.
