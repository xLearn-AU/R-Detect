import gradio as gr
from relative_tester import relative_tester

# from two_sample_tester import two_sample_tester
from utils import init_random_seeds

init_random_seeds()


def run_test(input_text):
    if not input_text:
        return "Now that you've built a demo, you'll probably want to share it with others. Gradio demos can be shared in two ways: using a temporary share link or permanent hosting on Spaces."
    # return two_sample_tester.test(input_text.strip())
    return relative_tester.test(input_text.strip())
    return f"Prediction: Human (Mocked for {input_text})"


# TODO: Add model selection in the future
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
#header { text-align: center; font-size: 3em; margin-bottom: 20px; color: #black; font-weight: bold;}
#output-text { font-weight: bold; font-size: 1.2em; border-radius: 10px; padding: 10px; background-color: #f4f4f4;}
.links {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-right: 10px;
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
    border: 2px soild #d1d1d1;
}

.output-text {
    flex: 1;  /* 1 part of the row */
    border-radius: 8px;
    padding: 12px;
    border: 2px soild #d1d1d1;
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
        gr.HTML('<div class="demo-banner">This is a demo. For the full version, please refer to the <a href="https://github.com/xLearn-AU/R-Detect" target="_blank">GitHub</a> or the <a href="https://openreview.net/forum?id=z9j7wctoGV" target="_blank">Paper</a>.</div>')

    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter Text Here",
            lines=8,
            elem_classes=["input-text"],  # Applying the CSS class
            value="Archetypes represent universal patterns that define certain events, objects, or people. In literature, they reflect common themes and ideas that resonate across cultures, helping people interpret and relate to stories. However, archetypes are not limited to literature; they also exist in everyday life and play a vital role in shaping how we perceive the world. Specifically, archetypes of people help us understand their character, actions, and motivations. In real life, we can identify certain types of individuals who share common traits shaped by both personal inclinations and their environments. For instance, a modern corporate leader like Elon Musk exemplifies the archetype of a hero or creator. His vision and courage to innovate make him a prime example of this archetype, as others often view him as someone driven by a desire to transform the world. These traits influence not only how he is perceived but also his actions and motivations. Recognizing archetypes in daily life allows for a deeper understanding of people and events. By identifying these characteristics, I can better grasp someone’s motivations and role, making their behavior more predictable. While real-life individuals may exhibit a blend of archetypal traits, the concept still helps me interpret their actions and personalities, which in turn informs how I interact with them. (This is a example from Chatgpt)"
        )
        output = gr.Textbox(
            label="Inference Result",
            placeholder="Made by Human or AI",
            elem_id="output-text",
            lines=8,
            elem_classes=["output-text"],
        )
    with gr.Row():
        # TODO: Add model selection in the future
        # model_name = gr.Dropdown(
        #     [
        #         "Faster Model",
        #         "Medium Model",
        #         "Powerful Model",
        #     ],
        #     label="Select Model",
        #     value="Medium Model",
        #     elem_classes=["select"],
        # )
        submit_button = gr.Button(
            "Run Detection", variant="primary", elem_classes=["button"]
        )
        clear_button = gr.Button("Clear", variant="secondary", elem_classes=["button"])

    submit_button.click(run_test, inputs=[input_text], outputs=output)
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
