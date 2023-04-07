from flask import Flask
from flask import request
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision
import PIL
import io
import json
import os
import numpy as np
import segmentation_models_pytorch


def create_app(config={}):
    app = Flask(__name__)
    CORS(app)
    app.config['SEVER_NAME'] = '0.0.0.0'
    app.app_context().push()
    return app

app = create_app()

@app.route('/', methods=['POST'])
def classify():
    # Get the image file from the request
    image_file = request.files['image']

    # Run the image through the classification model to get the prediction
    # step 1 load image as tensor
    image = FileStorage_to_Tensor(image_file)
    print(type(image))
    # step 2 segment image
    leaf, disease = segmentation_model(image.unsqueeze(0))
    # step 3 compute severity
    severity = compute_severity(leaf.squeeze(0), disease.squeeze(0))
    print('severity: ', severity)

    outputs = classification_model(disease)
    classification = get_classification(outputs)

    if not image_file:
        try:
            req_body = request.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if classification:
        # return func.HttpResponse(f".")
        return json.dumps({
                "classification": classification,
                "severity": severity
            }), 200 
    else:
        return "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.", 200


class SegmentationModel(nn.Module):
    def __init__(self, seg_path='models/mobilenetv2.3'):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(torch.cuda.get_device_name(self.device))
        else:
            self.device = None
            print('GPU is not available')
        # Define your segmentation model here
        # self.segmentation = models.segmentation.__dict__["fcn_resnet50"](pretrained=True)
        self.segmentation = torch.load(seg_path, map_location=torch.device('cpu'))
        # Freeze the segmentation layers
        for param in self.segmentation.parameters():
            param.requires_grad = False

    def forward(self, x):
        device = self.device
        # Forward pass through the segmentation model
        # x = x/255.0
        # resize
        x = torchvision.transforms.Resize(size=(512,512))(x)
        if x.shape[1] == 4:
          # if batch_size is 1
          if x.shape[0] == 1:
            img = x.squeeze(0)
            # transpose to shape: 512, 512, 4
            img = np.transpose(img, (1,2,0))
            pil_image = PIL.Image.fromarray(img.numpy(), 'RGBA')
            rgb_image = pil_image.convert('RGB')
            rgb_array = np.asarray(rgb_image, dtype=np.float32)
            x = torch.from_numpy(np.transpose(rgb_array, (2,1,0)))
            x = x.unsqueeze(0)
        x = x.to(device=device)
        # segment image first
        outputs = self.segmentation(x)
        # Apply softmax activation function to the output
        probs = torch.softmax(outputs, dim=1)
        # Get the predicted labels
        _, labels = torch.max(probs, dim=1)

        disease_mask = (labels == 2).float()
        disease_mask = torch.unsqueeze(disease_mask, 1)
        healthy_mask = (labels == 1).float()
        healthy_mask = torch.unsqueeze(healthy_mask, 1)
        leaf_mask = disease_mask + healthy_mask

        disease = x * disease_mask
        leaf = x * leaf_mask
        return (leaf, disease)

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # Define your classification model here
        self.classification = torchvision.models.resnet18(pretrained=True)
        # Replace the last layer with a new layer that has num_classes outputs
        num_features = self.classification.fc.in_features
        self.classification.fc = nn.Linear(num_features, num_classes)
        # self.label_dict = train_set.class_to_idx
    
    def forward(self, x):
        # Forward pass through the classification model
        x = self.classification(x)
        return x

'''
Labels dictionary
'''
label_dict = {
    'Bacterial Spot': 0, 
    'Early Blight': 1, 
    'Healthy': 2, 
    'Late Blight': 3,
    'Leaf Mold': 4, 
    'Septoria Leaf Spot': 5, 
    'Tomato Mosaic Virus': 6, 
    'Yellow Leaf Curl Virus': 7
    }

'''
Load segmentation model
'''
segmentation_model = SegmentationModel()

'''
Load classification model
'''
classification_model = ClassificationModel()
weights_path = 'models/classification-v3_stateDict'
classification_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
classification_model.train(mode=False)

'''
get predicted classification
'''
def get_classification(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    # this id may not be related to database ids
    predicted_class_id = torch.argmax(probabilities, dim=1)
    predicted_class = get_class_from_id(predicted_class_id.cpu().numpy()[0])
    return predicted_class

def get_class_from_id(id):
  label = list(label_dict.keys())[list(label_dict.values()).index(id)]
  return label

'''
/classify route receives uploaded image as a FileStorage object. 
This code converts it to a tensor so it can be sent to the model.
'''
def FileStorage_to_Tensor(file_storage_object):
    image_binary = file_storage_object.read()
    pil_image = PIL.Image.open(io.BytesIO(image_binary))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor_image = transform(pil_image)
    return tensor_image
    
'''
Calculate severity.
'''
def compute_severity(leaf, disease):
    # Check the shape of the tensor (should be in the format C x H x W)
    C, H, W = leaf.shape
    # Flatten the tensor to a 2D matrix
    image_flat = leaf.view(C, H * W)
    # Count the number of non-black pixels (i.e. pixels with at least one non-zero channel)
    leaf_pixels = torch.sum(torch.any(image_flat != 0, dim=0))
    # Print the number of non-black pixels
    print("Leaf Non Black Pixels:", leaf_pixels.item())

     # Check the shape of the tensor (should be in the format C x H x W)
    C, H, W = disease.shape
    # Flatten the tensor to a 2D matrix
    image_flat = disease.view(C, H * W)
    # Count the number of non-black pixels (i.e. pixels with at least one non-zero channel)
    disease_pixels = torch.sum(torch.any(image_flat != 0, dim=0))
    # Print the number of non-black pixels
    print("Disease Non Black Pixels:", disease_pixels.item())

    severity = disease_pixels.item() / (leaf_pixels.item() + disease_pixels.item())

    return severity * 100


def get_environment():
    environment = os.environ['AZURE_FUNCTIONS_ENVIRONMENT']

