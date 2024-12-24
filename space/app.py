import gradio as gr
from relative_tester import relative_tester


def detect_function(input_text):
    return relative_tester.test(input_text)


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Value")
    detect = gr.Button("Detect")
    detect.click(fn=detect_function, inputs=input_text, outputs=output)

demo.launch()
