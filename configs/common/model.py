import torch.nn as nn
from functools import partial
from detectron2.config import LazyCall as L
from modeling import DINOMattePromptDiverseV2,MattingCriterion,TransformerDecoder,TransformerDecoderLayer,Detail_Capture_DINO_V2

# Base
embed_dim, num_heads = 384, 6

model = L(DINOMattePromptDiverseV2)(
    embed_size = embed_dim,
    
    criterion=L(MattingCriterion)(
        losses = [ 'unknown_l1_loss','known_l1_loss', 'loss_pha_laplacian']
    ),
    pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
    pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],
    input_format = "RGB",
    size_divisibility=32,
    decoder=L(Detail_Capture_DINO_V2)(
        merge='concat',
        ),
    auxiliary='trimap',
    prompt_decoder=L(TransformerDecoder)(
        decoder_layer=L(TransformerDecoderLayer)(
            d_model=384,
            nhead=8), 
        num_layers=1,
        ),

)
model.decoder.img_chans=3


