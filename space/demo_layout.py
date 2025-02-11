import gradio as gr
import spaces


# TOKENIZER =
# MINIMUM_TOKENS = 64

# def count_tokens(text):
#     return len(TOKENIZER(text).input_ids)


# Mock function for testing layout
def run_test_power(model_name, real_text, generated_text, N=10):
    return f"Prediction: Human (Mocked for {model_name})"


# Change mode name
# def change_mode(mode):
#    if mode == "Faster Model":
#        .change_mode("t5-small")
#    elif mode == "Medium Model":
#        .change_mode("roberta-base-openai-detector")
#    elif mode == "Powerful Model":
#        .change_mode("falcon-rw-1b")
#    else:
#        gr.Error(f"Invaild mode selected.")
#    return mode


css = """
#header { text-align: center; font-size: 3em; margin-bottom: 20px; }
#output-text { font-weight: bold; font-size: 1.2em; }
.links { 
    display: flex; 
    justify-content: flex-end; 
    gap: 10px; 
    margin-right: 10px; 
    align-items: center;
}
.separator {
    margin: 0 5px;
    color: black;
}

/* Adjusting layout for Input Text and Inference Result */
.input-row {
    display: flex;
    width: 100%;
}

.input-text {
    flex: 3;  /* 4 parts of the row */
    margin-right: 1px;
}

.output-text {
    flex: 1;  /* 1 part of the row */
}

/* Set button widths to match the Select Model width */
.button {
    width: 250px;  /* Same as the select box width */
    height: 100px;  /* Button height */
}

/* Set height for the Select Model dropdown */
.select {
    height: 100px;  /* Set height to 100px */
}

/* Accordion Styling */
.accordion {
    width: 100%;  /* Set the width of the accordion to match the parent */
    max-height: 200px;  /* Set a max-height for accordion */
    overflow-y: auto;  /* Allow scrolling if the content exceeds max height */
    margin-bottom: 10px;  /* Add space below accordion */
    box-sizing: border-box;  /* Ensure padding is included in width/height */
}

/* Accordion content max-height */
.accordion-content {
    max-height: 200px;  /* Limit the height of the content */
    overflow-y: auto;  /* Add a scrollbar if content overflows */
}
"""

# Gradio App
with gr.Blocks(css=css) as app:
    with gr.Row():
        gr.HTML('<div id="header">R-detect On HuggingFace</div>')
    with gr.Row():
        gr.HTML(
            """
        <div class="links">
            <a href="https://openreview.net/forum?id=z9j7wctoGV" target="_blank">Paper</a>
            <span class="separator">|</span>
            <a href="https://github.com/xLearn-AU/R-Detect" target="_blank">Code</a>
            <span class="separator">|</span>
            <a href="mailto:1730421718@qq.com" target="_blank">Contact</a>
        </div>
        """
        )
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter Text Here",
            lines=8,
            elem_classes=["input-text"],  # Applying the CSS class
        )
        output = gr.Textbox(
            label="Inference Result",
            placeholder="Made by Human or AI",
            elem_id="output-text",
            lines=8,
            elem_classes=["output-text"],
        )
    with gr.Row():
        model_name = gr.Dropdown(
            [
                "Faster Model",
                "Medium Model",
                "Powerful Model",
            ],
            label="Select Model",
            value="Medium Model",
            elem_classes=["select"],
        )
        submit_button = gr.Button(
            "Run Detection", variant="primary", elem_classes=["button"]
        )
        clear_button = gr.Button("Clear", variant="secondary", elem_classes=["button"])

    submit_button.click(
        run_test_power, inputs=[model_name, input_text, input_text], outputs=output
    )
    clear_button.click(lambda: ("", ""), inputs=[], outputs=[input_text, output])

    with gr.Accordion("Disclaimer", open=False, elem_classes=["accordion"]):
        gr.Markdown(
            """
        - **Disclaimer**: This tool is for demonstration purposes only. It is not a foolproof AI detector.
        - **Accuracy**: Results may vary based on input length and quality.
        """
        )

    with gr.Accordion("Citations", open=False, elem_classes=["accordion"]):
        gr.Markdown(
            """
        ```
        @inproceedings{zhangs2024MMDMP,
            title={Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy},
            author={Zhang, Shuhai and Song, Yiliao and Yang, Jiahao and Li, Yuanqing and Han, Bo and Tan, Mingkui},
            booktitle = {International Conference on Learning Representations (ICLR)},
            year={2024}
        }
        ```
        """
        )

app.launch()
