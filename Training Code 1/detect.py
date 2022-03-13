from PIL import Image
from torchvision import transforms
import torchvision
import torch

model = torch.load("./BSD_model.pkl")

model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
input_image = Image.open("./test/18.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256), # RandomSizedCrop(224)??
    transforms.ToTensor(),
    normalize
    ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# Read the categories
categories = ["cat","dog"]
# Show top categories per image
top1_prob, top1_id = torch.topk(probabilities, 1)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

