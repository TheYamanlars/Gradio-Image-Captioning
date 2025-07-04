import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Adım 1: Model ve işlemciyi yükle
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Adım 2: Caption üreten fonksiyonu tanımla
def caption_image(input_image: np.ndarray):
    # Görseli NumPy'den PIL formatına çevir ve RGB yap
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Veriyi işlemciden geçir
    inputs = processor(images=raw_image, return_tensors="pt")
    
    # Modelden açıklama üret
    outputs = model.generate(**inputs, max_length=50)
    
    # Tokenları metne çevir
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# Adım 3: Gradio arayüzünü oluştur
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="numpy"),  # Gradio'dan NumPy olarak gelir
    outputs="text",
    title="Görsel Açıklama (BLIP Modeli)",
    description="Bu uygulama, yüklediğiniz görsele otomatik olarak açıklama üretir. BLIP modelini kullanır."
)

# Adım 4: Uygulamayı başlat
iface.launch()

# image.show()  #image'ı gormek icin