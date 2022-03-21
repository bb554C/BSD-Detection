from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import torchvision
import torch
from ShuffleNet2 import ShuffleNet2
import os

def detect_images(model, source, category):
    count_right = 0
    count_wrong = 0
    size = 256
    for filename in os.listdir(source):
        try:
            pic = os.path.join(source,filename)
            input_image = Image.open(pic)
            box = input_image.getbbox()

            if box[2] > box[3]:
                preprocess = transforms.Compose([transforms.CenterCrop(box[3]),
                                                 transforms.Resize(size),
                                                 transforms.ToTensor()])
            elif box[3] > box[2]:
                preprocess = transforms.Compose([transforms.CenterCrop(box[2]),
                                                 transforms.Resize(size),
                                                 transforms.ToTensor()])
            else:
                preprocess = transforms.Compose([transforms.Resize(size),
                                                 transforms.ToTensor()])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            else:
                input_batch = input_batch.to('cpu')
                model.to('cpu')
            with torch.no_grad():
                output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            categories = ["Healthy", "BlackSigatoka", "Unkown"]
            top_prob, top_id = torch.topk(probabilities, 1)
            if categories[top_id[0]] == category:
                count_right = count_right + 1
            else:
                #print(filename)
                #print(probabilities)
                #print(categories[top_id[0]], top_prob[0].item())
                count_wrong = count_wrong + 1
        except:
            print("ERROR: cant process",filename)
    return count_right, count_wrong
def countModel(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            count = count + 1
    return count

if __name__ == '__main__':
    test_folder = "test"
    num_classes = 2
    input_size = 256
    net_type = 2
    directory = os.path.dirname(os.path.realpath(__file__))
    modelFolder = os.path.join(directory, "Archived Models")
    BSD_folder = os.path.join(directory, test_folder,"BlackSigatoka") 
    Healthy_folder = os.path.join(directory, test_folder,"Healthy") 
    #Unkown_folder = os.path.join(directory, test_folder,)
    model_count = countModel(modelFolder)
    while model_count > 2:
        best_accuracy = 0
        worst_accuracy = 1
        most_accurate_model = ""
        worst_model = ""
        for filename in os.listdir(modelFolder):
            if filename.endswith(".pkl"):
                modelDir = os.path.join(modelFolder, filename)
                                   
                model = ShuffleNet2(2, 256, 2)
                model.load_state_dict(torch.load(modelDir))
                model.eval()

                count_healthy_right = 0
                count_blackSigatoka_right = 0
                count_Unkown_right = 0
                
                count_healthy_wrong = 0
                count_blackSigatoka_wrong = 0
                count_Unkown_wrong = 0
                
                count_healthy_right, count_healthy_wrong = detect_images(model, Healthy_folder, "Healthy")
                count_blackSigatoka_right, count_blackSigatoka_wrong = detect_images(model, BSD_folder, "BlackSigatoka")
                
                Health_Total = count_healthy_right + count_healthy_wrong
                BSD_Total = count_blackSigatoka_right + count_blackSigatoka_wrong
                Healthy_Accuracy = count_healthy_right / Health_Total
                BSD_Accuracy = count_blackSigatoka_right / BSD_Total
                print("MODEL NAME:",filename)
                print("Healthy - Correct:", count_healthy_right , "Wrong:", count_healthy_wrong, "Accuracy:", Healthy_Accuracy)
                print("BSD - Correct:", count_blackSigatoka_right , "Wrong:", count_blackSigatoka_wrong, "Accuracy:", BSD_Accuracy)

                Total_Accuracy = (count_healthy_right + count_blackSigatoka_right) / (Health_Total + BSD_Total)
                if best_accuracy < Total_Accuracy:
                    most_accurate_model = filename
                    best_accuracy = Total_Accuracy
                if worst_accuracy > Total_Accuracy:
                    worst_model = filename
                    worst_accuracy = Total_Accuracy
                print("Total Accuracy:",Total_Accuracy)
                Total_Specificity = count_blackSigatoka_right / BSD_Total
                print("Total Specificity:",Total_Specificity)
                Total_Sensitivity = count_healthy_right / Health_Total
                print("Total Sensitivity:",Total_Sensitivity)
                if Total_Accuracy < 0.90:
                    print("MODEL UNDER 90% ACCURACY",filename,"REMOVED")
                    os.remove(os.path.join(os.path.join(modelFolder,filename)))
                    worst_accuracy = 100
                    worst_model = ""
        print("BEST MODEL",most_accurate_model)
        if worst_model != most_accurate_model:
            try:
                os.remove(os.path.join(os.path.join(modelFolder,worst_model)))
                print("WORST MODEL",worst_model,"REMOVED")
            except:
                print("File does not exist")
        model_count = countModel(modelFolder)





