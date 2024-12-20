import gradio as gr
from two_sample_tester import TwoSampleTester

# Initialize the TwoSampleTester class, which will load the model, tokenizer, and other necessary components
tester = TwoSampleTester()


def detect_function(input_text, intensity):
    return tester.test(input_text)


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Value")
    detect = gr.Button("Detect")
    detect.click(fn=detect_function, inputs=input_text, outputs=output)

demo.launch()
