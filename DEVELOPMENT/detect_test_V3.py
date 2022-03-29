from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import torchvision
import torch
from ShuffleNet2 import ShuffleNet2
import os

def detect_images(model, source):
    count_Healthy = 0
    count_BSD = 0
    count_Unknown = 0
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

            categories = ["Healthy", "BlackSigatoka", "Unknown"]
            top_prob, top_id = torch.topk(probabilities, 1)
            if top_id[0] == 0:
                count_Healthy = count_Healthy + 1
            elif top_id[0] == 1:
                count_BSD = count_BSD + 1
            elif top_id[0] == 2:
                count_Unknown = count_Unknown + 1
        except:
            print("ERROR: cant process",filename)
    return count_Healthy, count_BSD, count_Unknown
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
    Unknown_folder = os.path.join(directory, test_folder, "Unknown")
    model_count = countModel(modelFolder)
    while model_count > 0:
        best_accuracy = 0
        worst_accuracy = 1
        most_accurate_model = ""
        worst_model = ""
        for filename in os.listdir(modelFolder):
            if filename.endswith(".pkl"):
                modelDir = os.path.join(modelFolder, filename)
                                   
                model = ShuffleNet2(4, 256, 2)
                model.load_state_dict(torch.load(modelDir))
                model.eval()

                Healthy = [0,0,0]
                BSD = [0,0,0]
                Unknown = [0,0,0]

                Healthy = detect_images(model, Healthy_folder)
                BSD = detect_images(model, BSD_folder)
                Unknown = detect_images(model, Unknown_folder)
                
                print(Healthy)
                print(BSD)
                print(Unknown)
                
                Health_Total = Healthy[0] + Healthy[1] + Healthy[2]
                BSD_Total = BSD[0] + BSD[1] + BSD[2]
                Unknown_Total = Unknown[0] + Unknown[1] + Unknown[2]

                Healthy_Accuracy = Healthy[0] / Health_Total
                BSD_Accuracy = BSD[1] / BSD_Total
                Unknown_Accuracy = Unknown[2] / Unknown_Total

                Healthy_Specificity = (BSD[1] + BSD[2] + Unknown[1] + Unknown[2]) / (BSD_Total + Unknown_Total)
                BSD_Specificity = (Healthy[0] + Healthy[2] + Unknown[0] + Unknown[2]) / (Health_Total + Unknown_Total)
                Unknown_Specificity = (Healthy[0] + Healthy[1] + BSD[0] + BSD[1]) / (Health_Total + BSD_Total)

                Healthy_Sensitivity = int(Healthy[0]) / Health_Total
                BSD_Sensitivity = int(BSD[1]) / BSD_Total
                Unknown_Sensitivity = int(Unknown[2]) / Unknown_Total
                
                print("MODEL NAME:",filename)
                print("Healthy:", Healthy ,
                      "Accuracy:", Healthy_Accuracy,
                      "Specificity:", Healthy_Specificity,
                      "Sensitivity", Healthy_Sensitivity)
                
                print("BSD:", BSD,
                        "Accuracy:", BSD_Accuracy,
                      "Specificity:", BSD_Specificity,
                      "Sensitivity", BSD_Sensitivity)
                
                print("Unknown:", Unknown ,
                      "Accuracy:", Unknown_Accuracy,
                      "Specificity:", Unknown_Specificity,
                      "Sensitivity", Unknown_Sensitivity)
                
                Total_Accuracy = (Healthy_Accuracy + BSD_Accuracy + Unknown_Accuracy) / 3
                Total_Specificity = (Healthy_Specificity + BSD_Specificity + Unknown_Specificity) / 3
                Total_Sensitivity = (Healthy_Sensitivity + BSD_Sensitivity + Unknown_Sensitivity) / 3

                if best_accuracy < Total_Accuracy:
                    most_accurate_model = filename
                    best_accuracy = Total_Accuracy
                if worst_accuracy > Total_Accuracy:
                    worst_model = filename
                    worst_accuracy = Total_Accuracy

                print("Total Accuracy:", Total_Accuracy)
                print("Total Specificity:", Total_Specificity)
                print("Total Sensitivity:", Total_Sensitivity)
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





