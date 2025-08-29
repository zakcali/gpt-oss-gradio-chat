# gpt-oss-gradio-chat
gradio chat interface using either openAI or Groq api

if you want to serve to LAN clients, then last line is:

    demo.launch(server_name="192.168.0.xx", server_port=7860) // set server_name="192.168.0.xx" according to your LAN ip

if you want to serve on your machine, then

    demo.launch()

if you want to serve to everybody connected to internet, then which creates an url for you
    
    demo.launch(share= True)

reference: https://www.gradio.app/guides/sharing-your-app 

