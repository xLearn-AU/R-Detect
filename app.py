import gradio as gr

from utils import init_random_seeds, config
from relative_tester import RelativeTester


def run_test(input_text):
    if not input_text:
        return "Please enter some text to test."
    return relative_tester.test(input_text.strip())
    return f"Prediction: Human (Mocked for {input_text})"


css = """
#header { text-align: center; margin-bottom: 5px; color: #black; font-weight: bold; font-weight: bold;}
#header_bigger { font-size: 2.5em; }
#header_smaller { font-size: 2em; }
.links {
    display: flex;
    justify-content: flex-end;
    gap: 5px;
    margin-right: 10px;
    margin-top: -10px;
    margin-bottom: -20px !important;
    align-items: center;
    font-size: 0.9em;
    color: #ADD8E6;
}
.separator {
    margin: 0 5px;
    color: #000;
}
/* Adjusting layout for Input Text and Inference Result */
.input-row {
    display: flex;
    width: 100%;
}
.input-text {
    flex: 3;  /* 4 parts of the row */
    margin-right: 1px;
    border-radius: 8px;
    padding: 12px;
    border: 2px solid #d1d1d1;
}
/* Set button widths to match the Select Model width */
.button {
    width: 250px;  /* Same as the select box width */
    height: 100px;  /* Button height */
    background-color: #ADD8E6;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.button:hover {
    background-color: #0000FF;
}
/* Set height for the Select Model dropdown */
.select {
    height: 100px;  /* Set height to 100px */
}
/* Accordion Styling */
.accordion {
    width: 100%;  /* Set the width of the accordion to match the parent */
    max-height: auto;  /* Set a auto-height for accordion */
    margin-bottom: 10px;  /* Add space below accordion */
    box-sizing: border-box;  /* Ensure padding is included in width/height */
}
/* Accordion content max-height */
.accordion-content {
    max-height: auto;  /* auto the height of the content */
}
.demo-banner {
    background-color: #f3f4f6;
    padding: 20px;
    border-radius: 10px;
    font-size: 1.1em;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
    color: #ff5722;
}
/* Green for Human text */
.highlighted-human {
    background-color: #d4edda;
    color: #155724;
    border: 2px solid #28a745;
}
/* Red for AI text */
.highlighted-ai {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #dc3545;
}
/* Yellow for errors */
.highlighted-error {
    background-color: #fff3cd;
    color: #856404;
    border: 2px solid #ffc107;
}
#output-text textarea {
    font-family: 'Impact', sans-serif;
}
"""

# Gradio App
with gr.Blocks(css=css) as app:
    with gr.Row():
        gr.HTML(
            '<div id="header"><span id="header_bigger">R-Detect: </span><span id="header_smaller">Human-Rewritten or AI-Generated</span></div>'
        )
    with gr.Row():
        gr.HTML(
            """
        <div class="links">
            <a href="https://openreview.net/forum?id=z9j7wctoGV" target="_blank">Paper</a>
            <span class="separator">|</span>
            <a href="https://github.com/xLearn-AU/R-Detect" target="_blank">Code</a>
            <span class="separator">|</span>
            <a href="mailto:yiliao.song@gmail.com" target="_blank">Contact</a>
        </div>
        """
        )

    with gr.Row():
        gr.HTML(
            '<div class="demo-banner">This is a demo running on CPU only. For the full version, please refer to the <a href="https://github.com/xLearn-AU/R-Detect" target="_blank">GitHub</a> or the <a href="https://openreview.net/forum?id=z9j7wctoGV" target="_blank">Paper</a>.</div>'
        )

    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter Text Here",
            lines=8,
            elem_classes=["input-text"],  # Applying the CSS class
            value="Hugging Face is a company and community that has become one of the leading platforms in the field of natural language processing (NLP). It is best known for developing and maintaining the Transformers library, which simplifies the use of state-of-the-art machine learning models for tasks such as text classification, language generation, translation, and more.",
        )
        output = gr.Textbox(
            label="Inference Result",
            placeholder="Made by Human or AI",
            elem_id="output-text",
            lines=2,  # Keep it compact
            interactive=False,  # Make it read-only
        )

    with gr.Row():
        submit_button = gr.Button(
            "Run Detection", variant="primary", elem_classes=["button"]
        )
        clear_button = gr.Button("Clear", variant="secondary", elem_classes=["button"])

    submit_button.click(run_test, inputs=[input_text], outputs=[output])
    clear_button.click(lambda: ("", ""), inputs=[], outputs=[input_text, output])

    with gr.Accordion("Disclaimer", open=True, elem_classes=["accordion"]):
        gr.Markdown(
            """
                - **Disclaimer**: This tool is for demonstration purposes only. It is not a foolproof AI detector.
                - **Accuracy**: Results may vary based on input length and quality.
            """
        )

    with gr.Accordion("Cite Our Work", open=True, elem_classes=["accordion"]):
        gr.Markdown(
            """
            ```
                @inproceedings{song2025deep,
                    title     = {Deep Kernel Relative Test for Machine-generated Text Detection},
                    author    = {Yiliao Song and Zhenqiao Yuan and Shuhai Zhang and Zhen Fang and Jun Yu and Feng Liu},
                    booktitle = {The Twelfth International Conference on Learning Representations},
                    year      = {2025},
                    url       = {https://openreview.net/pdf?id=z9j7wctoGV}
                }
            ```
            """
        )

    with gr.Accordion("Acknowledgement", open=True, elem_classes=["accordion"]):
        gr.Markdown(
            """
                We Thanks Jinqian Wang and Jerry Ye for their help in the development of this space.,
            """
        )


if __name__ == "__main__":
    config["use_gpu"] = False
    config["local_model"] = ""
    config["feature_ref_HWT"] = "./feature_ref_HWT_500.pt"
    config["feature_ref_MGT"] = "./feature_ref_MGT_500.pt"
    init_random_seeds()
    relative_tester = RelativeTester()
    app.launch()
