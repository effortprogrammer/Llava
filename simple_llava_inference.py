from Llava.VisionZip.visionzip.main import visionzip
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

print(dir(visionzip))
model_name = "NCSOFT/VARCO-VISION-14B-HF"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="float16",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
processor = AutoProcessor.from_pretrained(model_name)
device = model.device

model = visionzip(model, dominant=54, contextual=10)
print(model)
