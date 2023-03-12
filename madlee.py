import transformers

def startup():
    from diffusers import DiffusionPipeline
    import torch
    pipeline = DiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0",torch_dtype=torch.float16,safety_checker=None)
    pipeline.to("cuda")
    return pipeline

def generate(pipeline):
    prompt = input("Mad-lee: Inserisci un prompt per iniziare la generazione")
    image = pipeline(prompt).images[0]
    from IPython.display import display,Image
    image.save("generated.png",format="PNG",quality=100)
    display(Image("generated.png"))
