import gradio as gr
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser

import cv2

def adjust_resolution(image, mask):
    
    
    
    max_dimension = max(image.shape[0], image.shape[1])
    
    if max_dimension > 1920:
        scale_factor = 1280 / max_dimension
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        
        # 缩放图像
        resized_image = cv2.resize(image, (new_width, new_height))
        resized_mask = cv2.resize(mask, (new_width, new_height))
        return resized_image, resized_mask
    else:
        return image, mask


def determine_foreground_background(binary_map, similarity_map_self, average_similarity_self):
    # Compute mean similarity for the region labeled 1
    mean_similarity_1 = np.mean(similarity_map_self[binary_map == 1])
    
    # Compute mean similarity for the region labeled 0
    mean_similarity_0 = np.mean(similarity_map_self[binary_map == 0])
    
    # Compute the absolute distances to the average_similarity_self
    distance_1 = abs(mean_similarity_1 - average_similarity_self)
    distance_0 = abs(mean_similarity_0 - average_similarity_self)
    
    # Determine which region is foreground (set to 1) and which is background (set to 0)
    if distance_1 < distance_0:

        # Region with label 0 is the background
        binary_map=1-binary_map
    
    return binary_map

def generate_checkerboard_image(height, width, num_squares):
    num_squares_h = num_squares
    square_size_h = height // num_squares_h
    square_size_w = square_size_h
    num_squares_w = width // square_size_w
    

    new_height = num_squares_h * square_size_h
    new_width = num_squares_w * square_size_w
    image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(num_squares_h):
        for j in range(num_squares_w):
            start_x = j * square_size_w
            start_y = i * square_size_h
            color = 255 if (i + j) % 2 == 0 else 200
            image[start_y:start_y + square_size_h, start_x:start_x + square_size_w] = color

    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def get_data(image):
    images = image['image']
    mask = image['mask'][:,:,0].astype('bool').astype('float')

    images,mask = adjust_resolution(images,mask)

    input = {
            "image": torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)/255.0,
            "prompt": torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
            "trimap": torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
        }
    
    return input

if __name__=='__main__':

    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--device', default='cuda:0', type=str, required=True)
    
    args = parser.parse_args()
    device=args.device
    cfg = LazyConfig.load(args.config_dir)
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()
    DetectionCheckpointer(model).load(args.checkpoint_dir)
    
    def matting_inference(image):
        
        data = get_data(image)
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)
            output_ = model(data)
            output = output_['phas']
            input_x = image["image"]
            print(output.shape)
            output = torch.nn.functional.interpolate(output, [input_x.shape[0], input_x.shape[1]])
            output = output.flatten(0, 2)
            # output = output_['phas']
            output = (output*255).detach().cpu().numpy().astype('uint8')
            # h,w = model.features.shape[-2:]
            sim = (output_['sim']*255).squeeze().squeeze().detach().cpu().numpy().astype('uint8')
            sim_mask = (((sim>128).astype('float'))*255).astype('uint8')


            mask = ((model.prompt)*255).astype('uint8')
            background = generate_checkerboard_image(input_x.shape[0], input_x.shape[1], 8)
            foreground_alpha = input_x * np.expand_dims(output/255, axis=2).repeat(3,2)/255 + background * (1 - np.expand_dims(output/255, axis=2).repeat(3,2))/255
            foreground_alpha[foreground_alpha>1] = 1
            
        return output, mask, sim, sim_mask, foreground_alpha
    
    with gr.Blocks() as demo:
        
        gr.Markdown(
        "# <center>SMat</center>"
        )
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # input image
                image = gr.ImageMask(type='numpy')
                show_button = gr.Button(value='show') 
                with gr.Row():
                    alpha_matte = gr.outputs.Image(type='numpy')    
                    mask = gr.outputs.Image(type='numpy')    
                    sim = gr.outputs.Image(type='numpy')    
                    sim_mask = gr.outputs.Image(type='numpy')    
                    before = gr.outputs.Image(type='numpy')    

        
        show_button.click(
            matting_inference,
            [image],
            [alpha_matte, mask, sim, sim_mask, before]
        )

       
        
    demo.launch(share=True)
                
        

        