import random
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
# from models.vq.residual_vq import ResidualVQ
from models.vector_quantize_pytorch.residual_vq import ResidualVQ, GroupedResidualVQ
from models.vector_quantize_pytorch.residual_lfq import ResidualLFQ   
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        # self.quant = args.quantizer
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': 1, # args.num_quantizers,
            # 'shared_codebook': args.shared_codebook,
            'quantize_dropout': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'codebook_size': nb_code,
            'dim':code_dim,
            # 'codebook_dim' : 64
 
            # 'args': args,
        }
        print("RVQ!!!!!!!!!!!!!",rvqvae_config)
        # self.quantizer = ResidualVQ(**rvqvae_config,
        #                             # use_cosine_sim = True,

        #                             )
        # self.quantizer = ResidualLFQ(**rvqvae_config)
        self.quantizer = GroupedResidualVQ(**rvqvae_config,
                                           groups=8,
                                        #    kmeans_init = True,
                                        #    kmeans_iters = 10
                                           )

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        # import pdb;pdb.set_trace()
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = x_encoder.permute(0, 2, 1)
        # print(x_encoder.shape)
        _, code_idx, _,all_codes = self.quantizer(x_encoder, return_all_codes=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        # all_codes = all_codes
        all_codes = torch.cat([all_codes[0],all_codes[1]],dim=3)
        all_codes = all_codes.permute(1,3,2,0)
        return code_idx, all_codes

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in) #[batch,dim,seq]

        #~~~~~~~~~~~~~~~~~~~``
        x_encoder = x_encoder.permute(0, 2, 1) #torch.Size([32, 49, 512])
        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss = self.quantizer(x_encoder, 
                                                            # sample_codebook_temp=0.5
                                                            )

        x_quantized = x_quantized.permute(0, 2, 1) #torch.Size([32, 512, 49])
        # print(code_idx[0, :, 1])
        ## decoders
        x_out = self.decoder(x_quantized)
        perplexity = torch.tensor(1)
        commit_loss = torch.mean(commit_loss)
        # x_out = self.postprocess(x_decoder)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        # x = x.permute(0, 2, 1)
        # import pdb;pdb.set_trace()
        x_d = self.quantizer.get_codes_from_indices(torch.stack([x,x]))
        x_d = torch.cat([x_d[0],x_d[1]], dim=3).squeeze(0)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)
        # x = x.permute(1 ,0).unsqueeze(0)
        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)