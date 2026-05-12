import torch
import torch.nn as nn

from src.models.KP.models import MLP_LSTM_FE
from src.models.RGB.models import TSM_Resnet
from src.models.KP_RGB.models import TSM_Resnet_Encoder, MLP_FE, JointEncoders
from src.models.modules import FeatureMapExtractorModel, RNNHead, MLP

from typing import Tuple

from demo_KP.utils import get_vocab, getRuTokens
from src.utils import Decoder as CharDecoder

def load_kp_model(
        path_to_weights, cfg, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ) -> Tuple[MLP_FE, RNNHead]:
    fe = FeatureMapExtractorModel(num_keypoints=cfg.fe.num_keypoints, out_dim=cfg.fe.out_dim)
    mlp = MLP(
        input_dim=cfg.mlp.input_dim,
        hidden_dim=cfg.mlp.hidden_dim,
        output_dim=cfg.mlp.output_dim,
    )
    rnn_head = RNNHead(
        input_dim=cfg.decoder.input_dim,
        hidden_dim=cfg.decoder.hidden_dim,
        num_layers=cfg.decoder.num_layers,
        bidirectional=cfg.decoder.bidirectional,
        return_outs=cfg.decoder.return_outs,
        num_classes=cfg.decoder.num_classes,
    )
    model = MLP_LSTM_FE(fe=fe, mlp=mlp, decoder=rnn_head).to(device)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights, map_location=device))
    
    # encoder = KPEncoder(fe=fe, MLP=mlp)
    encoder = MLP_FE(mlp, fe)
    return encoder, model.decoder

def load_rgb_model(
        path_to_weights, cfg, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ) -> Tuple[TSM_Resnet_Encoder, RNNHead]:
    rnn_head = RNNHead(
        input_dim=cfg.decoder.input_dim,
        hidden_dim=cfg.decoder.hidden_dim,
        num_layers=cfg.decoder.num_layers,
        bidirectional=cfg.decoder.bidirectional,
        return_outs=cfg.decoder.return_outs,
        num_classes=cfg.decoder.num_classes,
    )
    chars = getRuTokens()
    _, _, char_list = get_vocab(chars)
    model = TSM_Resnet(
        "resnet34", CharDecoder(char_list), cfg.decoder.num_classes,
        decoder_net=rnn_head, unidirection=(not cfg.decoder.bidirectional)
    ).to(device)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights, map_location=device))

    encoder = TSM_Resnet_Encoder("resnet34", unidirection=(not cfg.decoder.bidirectional))
    encoder.backbone = model.backbone
    return encoder, model.decoder_net

class KPRGBEncoder(nn.Module):
    def __init__(
        self,
        encoder_rgb: TSM_Resnet_Encoder,
        encoder_kp: MLP_FE,
        reduction: str,
    ):
        """
        Initializes the JointEncoders module.

        Parameters
        ----------
        encoder_rgb : TSM_Resnet_Encoder
            An encoder for processing RGB data.

        encoder_kp : MLP_FE
            An encoder for processing keypoint data.
        
        reduction : str
            Specifies the method of combining the outputs of `encoder_rgb` and `encoder_kp`.
            Options are: "sum", "concat", "prod", "weight_sum", "weight_sum2".

        Attributes
        ----------
        encoder_rgb : TSM_Resnet_Encoder
            Stores the RGB encoder.

        encoder_kp : MLP_FE
            Stores the keypoint encoder.
        
        reduction : str
            Specifies how the encoded outputs are combined.

        weights1 : nn.Parameter
            Parameter used for weighted sum reduction.

        weights2 : nn.Parameter
            Parameter used for weighted sum2 reduction.
        """
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_kp = encoder_kp
        self.reduction = reduction
        self.weights1 = nn.Parameter(torch.randn(4, 1, 1024)) #.to(DEVICE)
        self.weights2 = nn.Parameter(torch.randn(4, 1, 1024)) #.to(DEVICE)

    def forward(self, x_rgb, x_kp, input_lenghts=None):
        """
        Forward pass through the joint encoders.

        Parameters
        ----------
        x_rgb : torch.Tensor
            Input RGB features.

        x_kp : torch.Tensor
            Input keypoint features.

        input_lenghts : list, optional
            List of sequence lengths for each sample in the batch.

        Returns
        -------
        torch.Tensor
            Output features after encoding and reduction.
        """
        if input_lenghts is not None:
            x_rgb = self.encoder_rgb(x_rgb, input_lenghts)
        else:
            x_rgb = self.encoder_rgb(x_rgb)

        x_kp = self.encoder_kp(x_kp)

        if self.reduction == "sum":
            x = x_kp + x_rgb
        elif self.reduction == "concat":
            x = torch.cat((x_rgb, x_kp), -1)
        elif self.reduction == "prod":
            x = x_rgb * x_kp
        elif self.reduction == "weight_sum":
            x = x_rgb * self.weights1 + x_kp * self.weights1
        elif self.reduction == "weight_sum2":
            x = x_rgb * self.weights1 + x_kp * self.weights2
        else:
            raise Exception("wrong reduction")
        
        return x

def load_kp_rgb_model(
        path_to_weights, cfg, reduction: str = 'sum',
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ) -> Tuple[KPRGBEncoder, RNNHead]:
    kp_enc, kp_dec = load_kp_model(None, cfg)
    rgb_enc, rgb_dec = load_rgb_model(None, cfg)
    
    chars = getRuTokens()
    _, _, char_list = get_vocab(chars)
    char_decoder = CharDecoder(char_list)
    model = JointEncoders(rgb_enc, kp_enc, char_decoder, rgb_dec, reduction=reduction).to(device)

    if path_to_weights is not None:
        model.load_state_dict(torch.load(path_to_weights, map_location=device))
    
    encoder = KPRGBEncoder(model.encoder_rgb, model.encoder_kp, model.reduction).to(device)
    return encoder, model.decoder_net
