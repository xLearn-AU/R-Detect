import gradio as gr
from two_sample_tester import TwoSampleTester


def detect_function(input_text, intensity):
    tester = TwoSampleTester(input_text)
    return tester.test()


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Value")
    detect = gr.Button("Detect")
    detect.click(fn=detect_function, inputs=input_text, outputs=output)

demo.launch()
