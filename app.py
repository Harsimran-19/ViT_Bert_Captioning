from transformers import VisionEncoderDecoderModel
from transformers import RobertaTokenizerFast
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import gradio as gr

MAX_LEN = 128  
tokenizer = RobertaTokenizerFast.from_pretrained('Byte_tokenizer', max_len=MAX_LEN)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
t = VisionEncoderDecoderModel.from_pretrained('Model')
title="Image to Caption Generator"
def predict(image):
    image=image.convert("RGB")
    caption=tokenizer.decode(t.generate(feature_extractor(image, return_tensors="pt").pixel_values)[0])
    caption=caption.replace("<s>", "").replace("</s>", "")
    return caption
demo = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="pil"),
    outputs=gr.outputs.Textbox(label="Generated Caption"),
    title=title,
)
demo.launch()
