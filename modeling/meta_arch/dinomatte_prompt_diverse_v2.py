import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from detectron2.structures import ImageList



class DINOMattePromptDiverseV2(nn.Module):
    def __init__(self,
                 *,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 auxiliary,
                 embed_size,
                 prompt_decoder=None,
                 ):
        super(DINOMattePromptDiverseV2, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.embedding_size = embed_size
        self.patch_size = 14
        self.criterion = criterion
        self.input_format = input_format
        self.auxiliary = auxiliary
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.prompt_cross_attn = prompt_decoder
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, trimap, prompts, targets, H, W, ori_images = self.preprocess_inputs(batched_inputs)

        
        batch_size = images.shape[0]
        mask_dim = (images.shape[2] / self.patch_size, images.shape[3] / self.patch_size) 
        x_ = self.backbone.forward_features(images)
        x = x_['x_norm_patchtokens']

        self.prompt = prompts[-1,:,:,:].bool().squeeze().detach().cpu().numpy()
        
        judge = torch.any((prompts.bool().view(batch_size,-1)),dim=-1,keepdim=True).int()
        cls_token = x_['x_norm_clstoken']
        prompts = F.interpolate(prompts,(int(mask_dim[0]),int(mask_dim[1])),mode='nearest')
        mask_token = prompts * (x.permute(0,2,1).reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1])))
        mask_token = torch.sum(mask_token, dim=[-1,-2])/(torch.sum(prompts, dim=[-1,-2])+1e-6)
        merged_token = torch.zeros_like(cls_token)
    
        for i in range(0,batch_size):
            if judge[i]==1:
                merged_token[i]=mask_token[i]
            else:
                merged_token[i]=cls_token[i]
        
                        
                
                    
        
        
        if self.prompt_cross_attn is not None:
            cls_token_cro = merged_token.unsqueeze(0)

            x_cro = x.permute(1,0,2)
            updated_cls_token = self.prompt_cross_attn(cls_token_cro, x_cro).permute(1,0,2)
            

        sim = torch.cosine_similarity(updated_cls_token, x, dim=-1)
        min_value = sim.min(dim=-1, keepdim=True).values
        max_value = sim.max(dim=-1, keepdim=True).values
        sim = (sim - min_value) / (max_value - min_value + 1e-6)
        sim_mask = sim>0.5
        
        
        sim = sim.reshape(batch_size,1,int(mask_dim[0]),int(mask_dim[1]))
        
        
        
        sim_mask = sim_mask.reshape(batch_size,1,int(mask_dim[0]),int(mask_dim[1]))
        
        self.sim=sim[-1,0,:,:].detach().cpu().numpy()
        self.sim_mask=sim_mask[-1,0,:,:].detach().cpu().numpy()
        self.image = ori_images[-1,:,:,:].permute(1,2,0).detach().cpu().numpy()
        
        
        x = x.permute(0,2,1)
        x = x.reshape(batch_size,self.embedding_size,int(mask_dim[0]),int(mask_dim[1]))

        
            
        outputs = self.decoder(x, sim, images)  
        outputs['sim_mask'] = sim_mask
        outputs['sim'] = sim
        if self.training:
            assert targets is not None
            self.per_image_prediction = (outputs['phas'][-1]*255).permute(1,2,0).detach().cpu().numpy().astype("uint8")
            self.per_image_gt = (targets['phas'][-1]*255).permute(1,2,0).detach().cpu().numpy().astype("uint8")
            sample_map = torch.zeros_like(trimap)
            sample_map[trimap>=0.5] = 1
            self.sample_map = sample_map[-1,0,:,:].detach().cpu().numpy()
            
            losses = self.criterion(sample_map ,outputs, targets)               
            return losses
        else:
            outputs['phas'] = outputs['phas'][:,:,:H,:W]
            outputs['sim'] = sim[-1,0,:,:]
            return outputs
        




    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        
        ori_images = images
        trimap = batched_inputs['trimap'].to(self.device)
        prompt = batched_inputs['prompt'].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std


        
            
        B, C, H, W = images.shape
        os=14
        if images.shape[-1]%os!=0 or images.shape[-2]%os!=0:
            new_H = (os-images.shape[-2]%os) + H
            new_W = (os-images.shape[-1]%os) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_prompts = torch.zeros((prompt.shape[0], prompt.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]
            new_prompts[:,:,:H,:W] = prompt[:,:,:,:]
            del images
            images = new_images
            del prompt
            prompt = new_prompts

        if "alpha" in batched_inputs:
            phas = batched_inputs["alpha"].to(self.device)
        else:
            phas = None
            
        return images, trimap, prompt, dict(phas=phas, trimap=trimap), H, W, ori_images