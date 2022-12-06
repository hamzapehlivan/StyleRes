import gradio as gr
from inference import initialize_styleres
from utils import AppUtils
from datasets.process_image import ImageProcessor
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='Which device to use')
args = parser.parse_args()

utils = AppUtils()
methods = utils.get_methods()
styleres = initialize_styleres('checkpoints/styleres_ffhq.pth', args.device)
image_processor = ImageProcessor('checkpoints/shape_predictor_68_face_landmarks.dat')

def process_image(image, method,  edit, factor, is_align_checked):
    cfg = utils.args_to_cfg(method, edit, factor)
    if is_align_checked:
        image = image_processor.align_face(image)
    image = image_processor.preprocess_image(image, is_batch=False)
    image = styleres.edit_images(image, cfg)
    image = image_processor.postprocess_image(image.detach().cpu().numpy(), is_batch=False)
    return image

def update_edit_dropdown(method):
    choices = utils.get_edits(method)
    return gr.Dropdown.update(choices=choices, value=choices[0])
    
def update_slider(method):
    minimum, maximum, step= utils.get_range(method)
    return gr.Slider.update(minimum=minimum, maximum=maximum, value=0, step=step, label=f"Strength [{minimum}, {maximum}]")


with gr.Blocks() as demo:
    gr.Markdown(
    """
    <h1 align="center">Transforming the Residuals for Real Image Editing with StyleGAN (CVPR2023)</h1>
    <center>
    <div> <a href="https://arxiv.org/abs/2212.14359">Paper</a> | <a href="https://github.com/hamzapehlivan/StyleRes">Github</a> | <a href="https://www.cs.bilkent.edu.tr/~adundar/projects/StyleRes/">Website</a></div> 
   </center>
    <center> Our model finds high-quality GAN space representations, which can also adopt to various edits without explicitly being trained on them.
            Here, we showcase some of the examples from the test dataset, as well as real internet images.  </center>
    """)
    with gr.Row():
        image_input = gr.Image(type="pil", shape=(256,256), label='Input Image', value="samples/demo_samples/116.jpg").style(height=256)
        image_output = gr.Image(type="pil", shape=(256,256), label='Output Image').style(height=256)

    with gr.Row():
        with gr.Column(scale=0.25, min_width=50):
            methods_drowdown = gr.Dropdown(methods, label="Choose Method", value=methods[0])
        with gr.Column(scale=0.25, min_width=50):
            edits_dropdown = gr.Dropdown(utils.get_edits(methods[0]), label="Choose Edit", value=utils.get_edits(methods[0])[0])
 

    with gr.Row(): 
        with gr.Column(scale=0.1, min_width=50):
            is_align_checked = gr.Checkbox(label="Crop + Align")
        with gr.Column(scale=0.4, min_width=50):
            factor_slider = gr.Slider(-5, 5, value=0, label="Strength [-5, 5]")
    with gr.Row(): 
        with gr.Column(scale=0.5, min_width=50):
            submit_button = gr.Button(value="Edit")
    
    gr.Examples(
        examples=utils.get_examples(),
        inputs=[image_input, methods_drowdown,  edits_dropdown, factor_slider, is_align_checked],
        outputs=image_output,
        fn=process_image,
        cache_examples=True,
    )
    methods_drowdown.change(update_edit_dropdown, inputs=methods_drowdown, outputs=edits_dropdown )
    methods_drowdown.change(update_slider, inputs=methods_drowdown, outputs=factor_slider)
    submit_button.click(process_image, inputs=[image_input, methods_drowdown,  edits_dropdown, factor_slider, is_align_checked], outputs=image_output)

demo.launch(debug=True)