from PIL import Image
from torchvision import transforms
import torchvision
import torch
from ShuffleNet2 import ShuffleNet2

model = ShuffleNet2()
model.load_state_dict(torch.load("./BSD_Model.pkl"))
model.eval()
input_image = Image.open("./test/5.jpg")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
categories = ["cat","dog", "unkown"]
top_prob, top_id = torch.topk(probabilities, 2)
for i in range(top_prob.size(0)):
    print(categories[top_id[i]], top_prob[i].item())

