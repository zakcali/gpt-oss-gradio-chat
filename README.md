# gpt-oss-gradio-chat
gradio chat interface using either openAI or Groq api

if you want to serve to LAN clients, then last line is:

    demo.launch(server_name="192.168.0.xx", server_port=7860) // set server_name="192.168.0.xx" according to your LAN ip

if you want to serve on your machine, then

    demo.launch()

if you want to serve to everybody connected to internet, then:
    
    demo.launch(share= True) //  creates an url for you

reference: https://www.gradio.app/guides/sharing-your-app 

"gpt-oss-gradio-Groq.py" works on every machine provided with an internet connection


"gpt-oss-gradio-openai.py" works on local machine. it works on my 4x RTX 3090 (96 GB VRAM) Linux machine by using command

    vllm serve openai/gpt-oss-120b --tensor-parallel-size 4 --async-scheduling

reference: https://huggingface.co/openai/gpt-oss-120b/discussions/122

