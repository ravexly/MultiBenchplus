"""Implements common fusion patterns."""

import torch
from torch import nn
from torch.nn import functional as F
import pdb
from torch.autograd import Variable



class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)



class ConcatEarly(nn.Module):
    """Concatenation of input data on dimension 2."""

    def __init__(self):
        """Initialize ConcatEarly Module."""
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of ConcatEarly.
        
        :param modalities: An iterable of modalities to combine
        """
        return torch.cat(modalities, dim=1)


# Stacking modalities
class Stack(nn.Module):
    """Stacks modalities together on dimension 1."""

    def __init__(self):
        """Initialize Stack Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        print(modalities)
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        print(torch.stack(flattened, dim=1).shape)
        return torch.stack(flattened, dim=2)


class ConcatWithLinear(nn.Module):
    """Concatenates input and applies a linear layer."""

    def __init__(self, input_dim, output_dim, concat_dim=1):
        """Initialize ConcatWithLinear Module.
        
        :param input_dim: The input dimension for the linear layer
        :param output_dim: The output dimension for the linear layer
        :concat_dim: The concatentation dimension for the modalities.
        """
        super(ConcatWithLinear, self).__init__()
        self.concat_dim = concat_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, modalities):
        """
        Forward Pass of Stack.
        
        :param modalities: An iterable of modalities to combine
        """
        return self.fc(torch.cat(modalities, dim=self.concat_dim))


class MultiplicativeInteractions3Modal(nn.Module):
    """Implements 3-Way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, task=None):
        """Initialize MultiplicativeInteractions3Modal object.

        :param input_dims: list or tuple of 3 integers indicating sizes of input
        :param output_dim: size of outputs
        :param task: Set to "affect" when working with social data.
        """
        super(MultiplicativeInteractions3Modal, self).__init__()
        self.a = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  [input_dims[2], output_dim], 'matrix3D')
        self.b = MultiplicativeInteractions2Modal([input_dims[0], input_dims[1]],
                                                  output_dim, 'matrix')
        self.task = task

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions3Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if self.task == 'affect':
            return torch.einsum('bm, bmp -> bp', modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])
        return torch.matmul(modalities[2], self.a(modalities[0:2])) + self.b(modalities[0:2])


class MultiplicativeInteractions2Modal(nn.Module):
    """Implements 2-way Modal Multiplicative Interactions."""
    
    def __init__(self, input_dims, output_dim, output, flatten=False, clip=None, grad_clip=None, flip=False):
        """
        :param input_dims: list or tuple of 2 integers indicating input dimensions of the 2 modalities
        :param output_dim: output dimension
        :param output: type of MI, options from 'matrix3D','matrix','vector','scalar'
        :param flatten: whether we need to flatten the input modalities
        :param clip: clip parameter values, None if no clip
        :param grad_clip: clip grad values, None if no clip
        :param flip: whether to swap the two input modalities in forward function or not
        
        """
        super(MultiplicativeInteractions2Modal, self).__init__()
        self.input_dims = input_dims
        self.clip = clip
        self.output_dim = output_dim
        self.output = output
        self.flatten = flatten
        if output == 'matrix3D':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                input_dims[0], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(
                input_dims[1], output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim[0], output_dim[1]))
            nn.init.xavier_normal(self.b)

        # most general Hypernetworks as Multiplicative Interactions.
        elif output == 'matrix':
            self.W = nn.Parameter(torch.Tensor(
                input_dims[0], input_dims[1], output_dim))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0], output_dim))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(input_dims[1], output_dim))
            nn.init.xavier_normal(self.V)
            self.b = nn.Parameter(torch.Tensor(output_dim))
            nn.init.normal_(self.b)
        # Diagonal Forms and Gating Mechanisms.
        elif output == 'vector':
            self.W = nn.Parameter(torch.Tensor(input_dims[0], input_dims[1]))
            nn.init.xavier_normal(self.W)
            self.U = nn.Parameter(torch.Tensor(
                self.input_dims[0], self.input_dims[1]))
            nn.init.xavier_normal(self.U)
            self.V = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(self.input_dims[1]))
            nn.init.normal_(self.b)
        # Scales and Biases.
        elif output == 'scalar':
            self.W = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.W)
            self.U = nn.Parameter(torch.Tensor(input_dims[0]))
            nn.init.normal_(self.U)
            self.V = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.V)
            self.b = nn.Parameter(torch.Tensor(1))
            nn.init.normal_(self.b)
        self.flip = flip
        if grad_clip is not None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(
                    grad, grad_clip[0], grad_clip[1]))

    def _repeatHorizontally(self, tensor, dim):
        return tensor.repeat(dim).view(dim, -1).transpose(0, 1)

    def forward(self, modalities):
        """
        Forward Pass of MultiplicativeInteractions2Modal.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]
        elif len(modalities) > 2:
            assert False
        m1 = modalities[0]
        m2 = modalities[1]
        if self.flip:
            m1 = modalities[1]
            m2 = modalities[0]

        if self.flatten:
            m1 = torch.flatten(m1, start_dim=1)
            m2 = torch.flatten(m2, start_dim=1)
        if self.clip is not None:
            m1 = torch.clip(m1, self.clip[0], self.clip[1])
            m2 = torch.clip(m2, self.clip[0], self.clip[1])

        if self.output == 'matrix3D':
            Wprime = torch.einsum('bn, nmpq -> bmpq', m1,
                                  self.W) + self.V  # bmpq
            bprime = torch.einsum('bn, npq -> bpq', m1,
                                  self.U) + self.b    # bpq
            output = torch.einsum('bm, bmpq -> bpq', m2,
                                  Wprime) + bprime   # bpq

        # Hypernetworks as Multiplicative Interactions.
        elif self.output == 'matrix':
            Wprime = torch.einsum('bn, nmd -> bmd', m1,
                                  self.W) + self.V      # bmd
            bprime = torch.matmul(m1, self.U) + self.b      # bmd
            output = torch.einsum('bm, bmd -> bd', m2,
                                  Wprime) + bprime             # bmd

        # Diagonal Forms and Gating Mechanisms.
        elif self.output == 'vector':
            Wprime = torch.matmul(m1, self.W) + self.V      # bm
            bprime = torch.matmul(m1, self.U) + self.b      # b
            output = Wprime*m2 + bprime             # bm

        # Scales and Biases.
        elif self.output == 'scalar':
            Wprime = torch.matmul(m1, self.W.unsqueeze(1)).squeeze(1) + self.V
            bprime = torch.matmul(m1, self.U.unsqueeze(1)).squeeze(1) + self.b
            output = self._repeatHorizontally(
                Wprime, self.input_dims[1]) * m2 + self._repeatHorizontally(bprime, self.input_dims[1])
        return output


class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    
    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        """
        Initialize LowRankTensorFusion object.
        
        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim+1, self.output_dim)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1, self.output_dim)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # init the fusion weights
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


class NLgate(torch.nn.Module):
    """
    Implements of Non-Local Gate-based Fusion.

    
    See section F4 of https://arxiv.org/pdf/1905.12681.pdf for details
    """
    
    def __init__(self, thw_dim, c_dim, tf_dim, q_linear=None, k_linear=None, v_linear=None):
        """
        q_linear, k_linear, v_linear are none if no linear layer applied before q,k,v.
        
        Otherwise, a tuple of (indim,outdim) is required for each of these 3 arguments.
        
        :param thw_dim: See paper
        :param c_dim: See paper
        :param tf_dim: See paper
        :param q_linear: See paper
        :param k_linear: See paper
        :param v_linear: See paper
        """
        super(NLgate, self).__init__()
        self.qli = None
        if q_linear is not None:
            self.qli = nn.Linear(q_linear[0], q_linear[1])
        self.kli = None
        if k_linear is not None:
            self.kli = nn.Linear(k_linear[0], k_linear[1])
        self.vli = None
        if v_linear is not None:
            self.vli = nn.Linear(v_linear[0], v_linear[1])
        self.thw_dim = thw_dim
        self.c_dim = c_dim
        self.tf_dim = tf_dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Apply Low-Rank TensorFusion to input.
        
        :param x: An iterable of modalities to combine. 
        """
        q = x[0]
        k = x[1]
        v = x[1]
        if self.qli is None:
            qin = q.view(-1, self.thw_dim, self.c_dim)
        else:
            qin = self.qli(q).view(-1, self.thw_dim, self.c_dim)
        if self.kli is None:
            kin = k.view(-1, self.c_dim, self.tf_dim)
        else:
            kin = self.kli(k).view(-1, self.c_dim, self.tf_dim)
        if self.vli is None:
            vin = v.view(-1, self.tf_dim, self.c_dim)
        else:
            vin = self.vli(v).view(-1, self.tf_dim, self.c_dim)
        matmulled = torch.matmul(qin, kin)
        finalout = torch.matmul(self.softmax(matmulled), vin)
        return torch.flatten(qin + finalout, 1)


class EarlyFusionTransformer(nn.Module):
    """Implements a Transformer with Early Fusion."""
    
    embed_dim = 9

    def __init__(self, n_features):
        """Initialize EarlyFusionTransformer Object.

        Args:
            n_features (int): Number of features in input.
        
        """
        super().__init__()

        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.linear = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        """Apply EarlyFusion with a Transformer Encoder to input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Layer Output
        """
     

        if isinstance(x, list):

            x = [torch.flatten(m, start_dim=1) for m in x]
            x = torch.cat(x, dim=1)
            x = x.unsqueeze(1)
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return self.linear(x)


class LateFusionTransformer(nn.Module):
    """Implements a Transformer with Late Fusion."""
    
    def __init__(self, embed_dim=9):
        """Initialize LateFusionTransformer Layer.

        Args:
            embed_dim (int, optional): Size of embedding layer. Defaults to 9.
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.conv = nn.Conv1d(
            1, self.embed_dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        """Apply LateFusionTransformer Layer to input.

        Args:
            x (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        if isinstance(x, list):
            # 将每个张量展平并拼接在一起
            x = [torch.flatten(m, start_dim=1) for m in x]
            x = torch.cat(x, dim=1)
        x = self.conv(x.view(x.size(0), 1, -1))
        x = x.permute(2, 0, 1)
        x = self.transformer(x)[-1]
        return x

class HierarchicalAttentionMultiToOne(nn.Module):
    def __init__(self, n_features, embed_dim=9):
        super().__init__()
        self.embed_dim = embed_dim
 
        self.modal_convs = nn.ModuleList([
            nn.Conv1d(f, embed_dim, kernel_size=1, bias=False) for f in n_features
        ])
        self.modal_transformers = nn.ModuleList([
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=3, batch_first=True), num_layers=2)
            for _ in n_features
        ])

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=3), num_layers=3
        )
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, modalities):

        encoded = []
        for i, mod in enumerate(modalities):
 
            mod = mod.unsqueeze(-1)  
            x = self.modal_convs[i](mod) 
            x = x.permute(2, 0, 1)         
            x = self.modal_transformers[i](x)[-1]  
            encoded.append(x.unsqueeze(0))  

        fused = torch.cat(encoded, dim=0) 
        fused = fused.permute(1, 0, 2)     
        fused = fused.permute(1, 0, 2)     
        out = self.fusion_transformer(fused)[-1] 
        return out


# class HierarchicalAttentionOneToMulti(nn.Module):
#     def __init__(self, n_modalities, feature_dim, embed_dim=9, num_heads=3, num_layers=2):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.n_modalities = n_modalities
        
#         # embed_dim
#         self.input_linears = nn.ModuleList([
#             nn.Linear(feature_dim, embed_dim) for _ in range(n_modalities)
#         ])
        
#         #  Transformer Encoder (one-stream)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#         self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         #  Transformer Encoders (multi-stream)
#         self.stream_transformers = nn.ModuleList([
#             nn.TransformerEncoder(encoder_layer, num_layers=num_layers) for _ in range(n_modalities)
#         ])
        
#         self.output_linear = nn.Linear(embed_dim * n_modalities, 1)
        
#     def forward(self, modalities):
#         # modalities: list,  [B, feature_dim]
#         batch_size = modalities[0].shape[0]
        
      
#         embeds = [self.input_linears[i](modalities[i]).unsqueeze(0) for i in range(self.n_modalities)]  
#         # embeds shape: [1, B, embed_dim]，变成序列的每个位置
        
#         x = torch.cat(embeds, dim=0)
        
#         x_shared = self.shared_transformer(x)
        
#         outputs = []
#         for i in range(self.n_modalities):
   
#             token_i = x_shared[i].unsqueeze(0)

#             out = self.stream_transformers[i](token_i)[-1]  # 取最后时间步
#             outputs.append(out)
            
     
#         fused = torch.cat(outputs, dim=-1)  # [B, embed_dim * n_modalities]
#         out = self.output_linear(fused)
#         return out

        
# class CrossAttentionFusion(nn.Module):
#     def __init__(self, feature_dim, embed_dim=9, num_heads=3):
#         super().__init__()
#         self.embed_dim = embed_dim

        
#         self.linear_q1 = nn.Linear(feature_dim, embed_dim)
#         self.linear_kv2 = nn.Linear(feature_dim, embed_dim)
        
#         self.linear_q2 = nn.Linear(feature_dim, embed_dim)
#         self.linear_kv1 = nn.Linear(feature_dim, embed_dim)

#         self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    
#         self.output_linear = nn.Linear(2 * embed_dim, 1)

#     def forward(self, modalities):
#         modality_1, modality_2 = modalities[0], modalities[1]  # [B, F]
#         B = modality_1.shape[0]

#          # ---- A attend B ----
#         q1 = self.linear_q1(modality_1).unsqueeze(1)   # [B, 1, embed_dim]
#         kv2 = self.linear_kv2(modality_2).unsqueeze(1)
#         attn1, _ = self.cross_attn_1(q1, kv2, kv2)      # [B, 1, embed_dim]
#         attn1 = attn1.squeeze(1)

#        # ---- B attend A ----
#         q2 = self.linear_q2(modality_2).unsqueeze(1)
#         kv1 = self.linear_kv1(modality_1).unsqueeze(1)
#         attn2, _ = self.cross_attn_2(q2, kv1, kv1)
#         attn2 = attn2.squeeze(1)


#         fusion = torch.cat([attn1, attn2], dim=-1)  # [B, 2 * embed_dim]
#         out = self.output_linear(fusion)            # [B, 1]
#         return out


# class CrossAttentionConcatFusion(nn.Module):
#     def __init__(self, feature_dim, embed_dim=9, num_heads=3):
#         super().__init__()
#         self.embed_dim = embed_dim


#         self.linear_q1 = nn.Linear(feature_dim, embed_dim)
#         self.linear_kv2 = nn.Linear(feature_dim, embed_dim)
        
#         self.linear_q2 = nn.Linear(feature_dim, embed_dim)
#         self.linear_kv1 = nn.Linear(feature_dim, embed_dim)

#         self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)


#         fusion_input_dim = 2 * embed_dim + 2 * feature_dim
#         self.fusion_linear = nn.Linear(fusion_input_dim, 1)

#     def forward(self, modalities):
#         modality_1, modality_2 = modalities[0], modalities[1]  # [B, F]

#         B = modality_1.shape[0]

#         # ---- A attend B ----
#         q1 = self.linear_q1(modality_1).unsqueeze(1)  # [B, 1, embed_dim]
#         kv2 = self.linear_kv2(modality_2).unsqueeze(1)
#         attn1, _ = self.cross_attn_1(q1, kv2, kv2)     # [B, 1, embed_dim]
#         attn1 = attn1.squeeze(1)                       # [B, embed_dim]

#         # ---- B attend A ----
#         q2 = self.linear_q2(modality_2).unsqueeze(1)
#         kv1 = self.linear_kv1(modality_1).unsqueeze(1)
#         attn2, _ = self.cross_attn_2(q2, kv1, kv1)
#         attn2 = attn2.squeeze(1)

     
#         fusion_input = torch.cat([modality_1, attn1, modality_2, attn2], dim=-1)  # [B, 2F + 2E]
#         out = self.fusion_linear(fusion_input)  # [B, 1]
#         return out

class HierarchicalAttentionOneToMulti(nn.Module):
    def __init__(self, n_features, embed_dim=9, num_heads=3, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = len(n_features)
        
        # embed_dim
        self.input_linears = nn.ModuleList([
            nn.Linear(f, embed_dim) for f in n_features
        ])
        
        #  Transformer Encoder (one-stream)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,batch_first=True)
        self.shared_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        #  Transformer Encoders (multi-stream)
        self.stream_transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers) for _ in range(self.n_modalities)
        ])
        
        self.output_linear = nn.Linear(embed_dim * self.n_modalities, 1)
        
    def forward(self, modalities):
        # modalities: list,  [B, feature_dim]
        batch_size = modalities[0].shape[0]
        
        embeds = [self.input_linears[i](modalities[i]).unsqueeze(0) for i in range(self.n_modalities)]  
     
        
        x = torch.cat(embeds, dim=0)
        
        x_shared = self.shared_transformer(x)
        
        outputs = []
        for i in range(self.n_modalities):
            token_i = x_shared[i].unsqueeze(0)
            out = self.stream_transformers[i](token_i)[-1] 
            outputs.append(out)
            
        fused = torch.cat(outputs, dim=-1)  # [B, embed_dim * n_modalities]
        out = self.output_linear(fused)
        return fused


class CrossAttentionFusion(nn.Module):
    def __init__(self, n_features, embed_dim=9, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = len(n_features)
        
        self.linear_q1 = nn.Linear(n_features[0], embed_dim)
        self.linear_kv2 = nn.Linear(n_features[1], embed_dim)
        
        self.linear_q2 = nn.Linear(n_features[1], embed_dim)
        self.linear_kv1 = nn.Linear(n_features[0], embed_dim)

        self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.output_linear = nn.Linear(2 * embed_dim, 1)

    def forward(self, modalities):
        modality_1, modality_2 = modalities[0], modalities[1]  # [B, F]
        B = modality_1.shape[0]

        # ---- A attend B ----
        q1 = self.linear_q1(modality_1).unsqueeze(1)   # [B, 1, embed_dim]
        kv2 = self.linear_kv2(modality_2).unsqueeze(1)
        attn1, _ = self.cross_attn_1(q1, kv2, kv2)      # [B, 1, embed_dim]
        attn1 = attn1.squeeze(1)

        # ---- B attend A ----
        q2 = self.linear_q2(modality_2).unsqueeze(1)
        kv1 = self.linear_kv1(modality_1).unsqueeze(1)
        attn2, _ = self.cross_attn_2(q2, kv1, kv1)
        attn2 = attn2.squeeze(1)

        fusion = torch.cat([attn1, attn2], dim=-1)  # [B, 2 * embed_dim]
        # out = self.output_linear(fusion)            # [B, 1]
        return fusion


class CrossAttentionConcatFusion(nn.Module):
    def __init__(self, n_features, embed_dim=9, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim


        self.linear_q1 = nn.Linear(n_features[0], embed_dim)
        self.linear_kv2 = nn.Linear(n_features[1], embed_dim)
        self.linear_q2 = nn.Linear(n_features[1], embed_dim)
        self.linear_kv1 = nn.Linear(n_features[0], embed_dim)


        self.cross_attn_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)


        fusion_input_dim = 4 * embed_dim  

   
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fusion_input_dim, nhead=num_heads),
            num_layers=1
        )

        self.fusion_linear = nn.Linear(fusion_input_dim, 1)

    def forward(self, modalities):
        modality_1, modality_2 = modalities[0], modalities[1]

   
        q1 = self.linear_q1(modality_1).unsqueeze(1)  # [B, 1, E]
        kv2 = self.linear_kv2(modality_2).unsqueeze(1)  # [B, 1, E]
        attn1, _ = self.cross_attn_1(q1, kv2, kv2)  # [B, 1, E]
        attn1 = attn1.squeeze(1)  # [B, E]


        q2 = self.linear_q2(modality_2).unsqueeze(1)  # [B, 1, E]
        kv1 = self.linear_kv1(modality_1).unsqueeze(1)  # [B, 1, E]
        attn2, _ = self.cross_attn_2(q2, kv1, kv1)  # [B, 1, E]
        attn2 = attn2.squeeze(1)  # [B, E]

        # 拼接所有特征
        fusion_input = torch.cat([self.linear_q1(modality_1), attn1, self.linear_q2(modality_2), attn2], dim=-1)  # [B, 4 * embed_dim]
        fusion_input = fusion_input.unsqueeze(1)  # [B, 1, 4 * embed_dim]


        fused = self.fusion_transformer(fusion_input).squeeze(1)  # [B, 4 * embed_dim]

    
        out = self.fusion_linear(fused)  # [B, 1]
        return fused


class MultiModalCrossAttentionFusion(nn.Module):
    def __init__(self, n_features, embed_dim=9, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_modalities = len(n_features)


        self.linears_q = nn.ModuleList([nn.Linear(n_features[i], embed_dim) for i in range(self.n_modalities)])
        self.linears_kv = nn.ModuleList([nn.Linear(n_features[i], embed_dim) for i in range(self.n_modalities)])


        self.cross_attns = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) for _ in range(self.n_modalities)])


        self.fusion_linear = nn.Linear(self.n_modalities * embed_dim, 1)

    def forward(self, modalities):
        B = modalities[0].shape[0]

  
        attns = []
        for i in range(self.n_modalities):
            q = self.linears_q[i](modalities[i]).unsqueeze(1)  # [B, 1, embed_dim]
            kv = []
            for j in range(self.n_modalities):
                if i != j:
                    kv.append(self.linears_kv[j](modalities[j]).unsqueeze(1))  # [B, 1, embed_dim]
            kv = torch.cat(kv, dim=1)  # [B, n_modalities - 1, embed_dim]
            attn, _ = self.cross_attns[i](q, kv, kv)  # [B, 1, embed_dim]
            attn = attn.squeeze(1)  # [B, embed_dim]
            attns.append(attn)

 
        fusion = torch.cat(attns, dim=-1)  # [B, n_modalities * embed_dim]
        out = self.fusion_linear(fusion)  # [B, 1]
    
        return fusion


class MultiModalCrossAttentionConcatFusion(nn.Module):
    def __init__(self, n_features, embed_dim=9, num_heads=3, projection=False):
        super().__init__()
        self.projection = projection
        self.embed_dim = embed_dim
        self.n_modalities = len(n_features)


        self.linear_qs = nn.ModuleList([
            nn.Linear(f_dim, embed_dim) for f_dim in n_features
        ])
        self.linear_kvs = nn.ModuleList([
            nn.Linear(f_dim, embed_dim) for f_dim in n_features
        ])

  
        self.cross_attn_layers = nn.ModuleDict()
        for i in range(self.n_modalities):
            for j in range(self.n_modalities):
                if i != j:
                    key = f"{i}_attend_{j}"
                    self.cross_attn_layers[key] = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

   
        fusion_input_dim = self.n_modalities * embed_dim + self.n_modalities * (self.n_modalities - 1) * embed_dim

        projected_dim = 1536
        self.fusion_projection = nn.Linear(fusion_input_dim, projected_dim)
        if projection:
            fusion_input_dim = projected_dim

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fusion_input_dim, nhead=3),
            num_layers=2
        )
        self.fusion_linear = nn.Linear(fusion_input_dim, 1)

    def forward(self, modalities):  # modalities: list of [B, Fi]
        B = modalities[0].shape[0]

   
        all_features = []

        for i in range(self.n_modalities):
  
            orig = modalities[i]  # [B, Fi]
            orig_embed = self.linear_qs[i](orig)  # [B, E]
            all_features.append(orig_embed)

            q_i = orig_embed.unsqueeze(1)  # [B, 1, E]

            for j in range(self.n_modalities):
                if i == j:
                    continue
                kv_j = self.linear_kvs[j](modalities[j]).unsqueeze(1)  # [B, 1, E]
                attn_layer = self.cross_attn_layers[f"{i}_attend_{j}"]
                attn_out, _ = attn_layer(q_i, kv_j, kv_j)  # [B, 1, E]
                all_features.append(attn_out.squeeze(1))  # [B, E]


        fusion_input = torch.cat(all_features, dim=-1)  # [B, total_dim]
        fusion_input = fusion_input.unsqueeze(1)        # [B, 1, total_dim]
        if self.projection:
            fusion_input = self.fusion_projection(fusion_input)

        fused = self.fusion_transformer(fusion_input).squeeze(1)  # [B, total_dim]


        out = self.fusion_linear(fused)  # [B, 1]
        return fused