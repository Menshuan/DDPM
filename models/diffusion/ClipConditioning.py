import torch
import clip

class ClipConditioning:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)

    def get_text_condition(self, text):
        # 获取文本的CLIP编码
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        condition = {'text_encoding': text_features}
        return condition

    def get_image_condition(self, image):
        # 获取图像的CLIP编码
        image_preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_preprocessed)
        condition = {'image_encoding': image_features}
        return condition