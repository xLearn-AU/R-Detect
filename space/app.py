import gradio as gr
from relative_tester import RelativeTester
from model_loader import ModelLoader

model_loader = ModelLoader()
model, tokenizer, DEVICE = model_loader.model, model_loader.tokenizer
tester = RelativeTester(model, tokenizer)


def detect_function(input_text):
    return tester.test(input_text)


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Value")
    detect = gr.Button("Detect")
    detect.click(fn=detect_function, inputs=input_text, outputs=output)

demo.launch()
