import gradio as gr
from relative_tester import relative_tester
from two_sample_tester import two_sample_tester
from utils import init_random_seeds

init_random_seeds()


def detect_function(input_text):
    if not input_text:
        return "Now that you've built a demo, you'll probably want to share it with others. Gradio demos can be shared in two ways: using a temporary share link or permanent hosting on Spaces."
    # return two_sample_tester.test(input_text.strip())
    return relative_tester.test(input_text.strip())


with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text")
    output = gr.Textbox(label="Output Value")
    detect = gr.Button("Detect")
    detect.click(fn=detect_function, inputs=input_text, outputs=output)

demo.launch()
