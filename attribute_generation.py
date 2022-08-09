import os
from argparse import Namespace
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip
import sys
from tqdm import tqdm
sys.path.append(".")
sys.path.append("..")
from utils.common import tensor2im

from models.e4e import e4e
# from mapper.hairclip_mapper2 import HairCLIPMapper

def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image

def run_alignment(image_path):
    import dlib
    from scripts.align_faces_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

# captions_dict = {"Blond_Hair" : "This person has blond hair.",
#                  "Bushy_Eyebrows" : "This person has bushy eyebrows.",
#                  "Chubby" : "This person is chubby.",
#                  "Double_Chin" : "This person has double chin.",
#                  "Eyeglasses" : "This person has eyeglasses.",
#                  "Goatee" : "This person has a goatee.",
#                  "Gray_Hair" : "This person has gray hair.",
#                  "Heavy_Makeup" : "This person wears heavy makeup.",
#                  "Male" : "This is a male.",
#                  "Mouth_Slightly_Open" : "This person has mouth slightly open.",
#                  "Mustache" : "This person has a mustache.",
#                  "Rosy_Cheeks" : "This person has rosy cheeks.",
#                  "Smiling" : "This person is smiling.",
#                  "Wearing_Lipstick" : "This person is wearing lipstick.",
#                  "Wearing_Necktie" : "This person is wearing a necktie."}

# captions_dict = {"Blond_Hair" : {"female" : "She has blond hair.", "male" : "He has blond hair."},
#                  "Bushy_Eyebrows" : {"female" : "She has bushy eyebrows.", "male" : "He has bushy eyebrows."},
#                  "Chubby" : {"female" : "She is chubby.", "male" : "He is chubby."},
#                  "Double_Chin" : {"female" : "She has double chin.", "male" : "He has double chin."},
#                  "Eyeglasses" : {"female" : "She has eyeglasses.", "male" : "He has eyeglasses."},
#                  "Goatee" : {"female" : "He has goatee.", "male" : "He has goatee."},
#                  "Gray_Hair" : {"female" : "She has gray hair.", "male" : "He has gray hair."},
#                  "Heavy_Makeup" : {"female" : "She wears heavy makeup.", "male" : "She wears heavy makeup."},
#                  "Male" : {"female" : "This is a male.", "male" : "This is a male."},
#                  "Mouth_Slightly_Open" : {"female" : "She has mouth slightly open.", "male" : "He has mouth slightly open."},
#                  "Mustache" : {"female" : "He has mustache.", "male" : "He has mustache."},
#                  "Rosy_Cheeks" : {"female" : "She has rosy cheeks.", "male" : "He has rosy cheeks."},
#                  "Smiling" : {"female" : "She is smiling.", "male" : "He is smiling."},
#                  "Wearing_Lipstick" : {"female" : "She is wearing lipstick.", "male" : "She is wearing lipstick."},
#                  "Wearing_Necktie" : {"female" : "She is wearing necktie.", "male" : "He is wearing necktie."}}

captions_dict = {"Blond_Hair" : {"female" : "She has blond hair.", "male" : "He has blond hair."},
                 "Bushy_Eyebrows" : {"female" : "She has bushy eyebrows.", "male" : "He has bushy eyebrows."},
                 "Chubby" : {"female" : "She is chubby.", "male" : "He is chubby."},
                 "Double_Chin" : {"female" : "He has double chin.", "male" : "He has double chin."}, # changed
                 "Eyeglasses" : {"female" : "She has eyeglasses.", "male" : "He has eyeglasses."},
                 "Goatee" : {"female" : "He has goatee.", "male" : "He has goatee."},
                 "Gray_Hair" : {"female" : "She has gray hair.", "male" : "He has gray hair."},
                 "Heavy_Makeup" : {"female" : "She wears heavy makeup.", "male" : "She wears heavy makeup."},
                 "Male" : {"female" : "This is a male.", "male" : "This is a male."},
                 "Mouth_Slightly_Open" : {"female" : "She has mouth slightly open.", "male" : "He has mouth slightly open."},
                 "Mustache" : {"female" : "He has mustache.", "male" : "He has mustache."},
                 "Rosy_Cheeks" : {"female" : "She has rosy cheeks.", "male" : "She has rosy cheeks."}, # changed
                 "Smiling" : {"female" : "She is smiling.", "male" : "He is smiling."},
                 "Wearing_Lipstick" : {"female" : "She is wearing lipstick.", "male" : "She is wearing lipstick."},
                 "Wearing_Necktie" : {"female" : "He is wearing necktie.", "male" : "He is wearing necktie."}} # changed



label_path = "/datasets/CelebA/Anno/list_attr_celeba.txt"
label_list = open(label_path).readlines()[2:]
data_label = []
for i in range(len(label_list)):
    data_label.append(label_list[i].split())

# transform label into 0 and 1
for m in range(len(data_label)):
    data_label[m] = [n.replace('-1', '0') for n in data_label[m][1:]]
    data_label[m] = [int(p) for p in data_label[m]]
    
gender_index = 20

# captions_path = "generated_captions.txt"
# f1 = open(captions_path, "r")
inference_images_path = "/scratch/users/abaykal20/LACE/FFHQ/prepare_models_data/label_indexes"
images_path = "/datasets/CelebA/Img/img_align_celeba/"

EXPERIMENT_DATA_ARGS = {
    "celeba_encode": {
        "model_path": "exp_text_augment_consistency2/checkpoints/best_model.pt",
        "e4e_path": "/scratch/users/abaykal20/sam/SAM/pretrained_models/e4e_ffhq_encode.pt",
        "transform": transforms.Compose([
            # transforms.CenterCrop((178,178)),
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS["celeba_encode"]
# tf2 = transforms.Compose([
#             transforms.CenterCrop((178,178)),
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

print("Loading Models")
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
net = e4e(opts)
net.eval()
net.cuda()

print("Model Succesfully Loaded!")

clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda')
print("Starting inference")

exp_name = "3_step/"
num_text_iters = 3
if not os.path.isdir("aligned_attribute_classification/augment/" + exp_name):
    os.mkdir("aligned_attribute_classification/augment/" + exp_name)

img_transforms = EXPERIMENT_ARGS['transform']
for key, value in captions_dict.items():
    inference_path = os.path.join(inference_images_path, key + ".txt")
    f2 = open(inference_path, "r")
    # custom_caption = value
    if not os.path.isdir("aligned_attribute_classification/augment/" + exp_name + key):
        os.mkdir("aligned_attribute_classification/augment/" + exp_name + key)
    # if not os.path.isdir("metric_images/" + key):
    #     os.mkdir("metric_images/" + key)
    print("Generating", key)
    for i in tqdm(range(50)):
        img_idx = f2.readline().rstrip()
        img_idx_int = int(img_idx.lstrip("0").rstrip(".jpg")) - 1
        if data_label[img_idx_int][gender_index] == 0:
            gender = "female"
        else:
            gender = "male"
        custom_caption = value[gender]
        # complete_image_path = os.path.join(images_path, img_idx)
        complete_image_path = "/scratch/users/abaykal20/restyle-encoder/metric_images/" + key + "/{}.jpg".format(i)
        original_image = Image.open(complete_image_path).convert("RGB")
        # aligned_image = run_alignment(complete_image_path)
        # if aligned_image is None:
        #     input_image = tf2(original_image)
        # else:
        #     input_image = img_transforms(aligned_image)
        # tensor2im(input_image).save("metric_images/" + key + "/" + "{}.jpg".format(i))
        input_image = img_transforms(original_image)
        input_image = input_image.unsqueeze(0)
        text_input = clip.tokenize(custom_caption)
        text_input = text_input.cuda()
        input_image = input_image.cuda().float()
        with torch.no_grad():
            avg_image = get_avg_image(net)
            text_features = clip_model.encode_text(text_input).float()

            y_hat, latent = None, None
            for iter in range(5):
                if iter == 0:
                    avg_image_for_batch = avg_image.unsqueeze(0).repeat(input_image.shape[0], 1, 1, 1)
                    x_input = torch.cat([input_image, avg_image_for_batch], dim=1)
                else:
                    x_input = torch.cat([input_image, y_hat], dim=1)
                
                y_hat, latent = net.forward(x_input,
                                            latent=latent,
                                            randomize_noise=False,
                                            return_latents=True,
                                            resize=True)
            
            for iter in range(num_text_iters):
                x_input = torch.cat([input_image, y_hat], dim=1)
                rsz = True
                if iter == num_text_iters - 1:
                    rsz = False
                y_hat, latent = net.forward_text(x_input, 
                                                 text_features, 
                                                 latent=latent, 
                                                 randomize_noise=False, 
                                                 return_latents=True, 
                                                 resize=rsz)
            

            # result_tensor, _ = mapper.decoder([w_hat], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1)
            result_tensor = y_hat.squeeze(0)
            result_image = tensor2im(result_tensor)


            save_path = "aligned_attribute_classification/augment/" + exp_name + key + "/"
            save_path = save_path + "{}.jpg".format(i)
            result_image.save(save_path)
    f2.close()
            

print("Finished inference")