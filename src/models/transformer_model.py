import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src.data import utils # 确保 utils 在你的PYTHONPATH中或者可以通过相对路径访问
from src.frameworks import diffusion_utils # 同样，确保可访问
from src.models.layers import Xtoy, Etoy, masked_softmax # 同样


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx, **kwargs) # Added **kwargs
        self.k = Linear(dx, dx, **kwargs) # Added **kwargs
        self.v = Linear(dx, dx, **kwargs) # Added **kwargs

        # FiLM E to X
        self.e_add = Linear(de, dx, **kwargs) # Added **kwargs
        self.e_mul = Linear(de, dx, **kwargs) # Added **kwargs

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx, **kwargs)           # Warning: here it's dx and not de. Added **kwargs
        self.y_e_add = Linear(dy, dx, **kwargs)           # Added **kwargs

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx, **kwargs)           # Added **kwargs
        self.y_x_add = Linear(dy, dx, **kwargs)           # Added **kwargs

        # Process y
        self.y_y = Linear(dy, dy, **kwargs)               # Added **kwargs
        self.x_y = Xtoy(dx, dy)      # Assuming Xtoy, Etoy handle device/dtype internally or don't need **kwargs
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx, **kwargs)             # Added **kwargs
        self.e_out = Linear(dx, de, **kwargs)             # Added **kwargs
        # Using a new Sequential for y_out that matches the other MLP structures more closely
        self.y_out = nn.Sequential(nn.Linear(dy, dy, **kwargs), nn.ReLU(), nn.Linear(dy, dy, **kwargs)) # Added **kwargs


    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        # ---- MODIFICATION START (Ensuring diffusion_utils is accessible if used) ----
        # Assuming diffusion_utils.assert_correctly_masked exists and is imported
        if 'diffusion_utils' in globals() and hasattr(diffusion_utils, 'assert_correctly_masked'):
            diffusion_utils.assert_correctly_masked(Q, x_mask)
        # ---- MODIFICATION END ----
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, n, 1, n_head, df) -> PyTorch broadcasts, original was (bs,1,n,...)
        K = K.unsqueeze(1)                              # (bs, 1, n, n_head, df) -> Original was (bs,n,1,...)
                                                        # My previous versions were Q:(bs,1,n,...) and K:(bs,n,1,...) which is more standard for Q*K.T like ops
                                                        # Let's keep original Q:(bs,n,1,...) K:(bs,1,n,...) for Y = Q*K if Y is (bs, n, n, ...)
                                                        # Oh, Q.unsqueeze(2) gives (bs, n, 1, n_head, df)
                                                        # K.unsqueeze(1) gives (bs, 1, n, n_head, df)
                                                        # Then Y = Q * K is (bs, n, n, n_head, df) via broadcasting. This is correct.

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K 
        Y = Y / math.sqrt(Y.size(-1)) # df
        # ---- MODIFICATION START ----
        if 'diffusion_utils' in globals() and hasattr(diffusion_utils, 'assert_correctly_masked'):
            diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1).expand_as(Y)) # expand_as(Y)
        # ---- MODIFICATION END ----

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE_intermediate = Y.flatten(start_dim=3)                      # bs, n, n, dx (dx = n_head * df)
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)                 # bs, 1, 1, dx 
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)                 # bs, 1, 1, dx
        newE_intermediate = ye1 + (ye2 + 1) * newE_intermediate

        # Output E
        newE = self.e_out(newE_intermediate) * e_mask1 * e_mask2      # bs, n, n, de
        # ---- MODIFICATION START ----
        if 'diffusion_utils' in globals() and hasattr(diffusion_utils, 'assert_correctly_masked'):
            diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)
        # ---- MODIFICATION END ----

        # Compute attentions. attn is still (bs, n, n, n_head, df) but Y was (bs, n, n, n_head, df)
        # softmax_mask should be (bs, n, n, n_head)
        # Original was: softmax_mask = e_mask2.expand(-1, n, -1, self.n_head) # bs, 1, n, n_head -> bs, n, n, n_head
        # This seems to mask rows of K. Softmax is over dim=2 (the K dimension, or source nodes)
        # Correct mask for softmax over dim 2 (keys for each query):
        # Y shape (bs, n_queries, n_keys, n_head, df_qk)
        # softmax_mask for Y needs to be (bs, n_queries, n_keys, n_head)
        # node_mask is (bs, n). x_mask is (bs, n, 1). e_mask1 (bs, n, 1, 1), e_mask2 (bs, 1, n, 1)
        # The mask for attention weights A_ij (query i, key j) should depend on whether key j is a real node.
        # So, it should be based on e_mask2 (bs, 1, n, 1), expanded for queries and heads.
        attn_mask_for_softmax = e_mask2.squeeze(-1).expand(-1, n, -1) # (bs, n_queries, n_keys)
        attn_mask_for_softmax = attn_mask_for_softmax.unsqueeze(-1).expand(-1, -1, -1, self.n_head) # (bs, n, n, n_head)

        # Y already had df_qk dim, so let's assume Y is (bs, n_q, n_k, n_head) before softmax
        # The original code applied softmax to Y which was (bs, n, n, n_head, df). This seems unusual.
        # Typically, softmax is over a score that's (bs, n_q, n_k, n_head).
        # Let's assume Y needs to be summed over the `df` dimension or only one `df` feature is used for attention score.
        # Given Y = Q*K / sqrt(df) + E-terms, it's likely Y itself is the score.
        # If Y is (bs, n, n, n_head, df), then softmax(Y, dim=2) would be over keys for each (query, head, feature_in_df).
        # This is plausible. Let's stick to the original structure as much as possible.

        attn = masked_softmax(Y, attn_mask_for_softmax.unsqueeze(-1).expand_as(Y), dim=2)  # softmax over n_keys, for each query, head, and feature_dim_df

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        # attn is (bs, n, n, n_head, df), V is (bs, 1, n, n_head, df)
        # Broadcasting: attn * V will be (bs, n_queries, n_keys, n_head, df_v)
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2) # Sum over keys --> (bs, n_queries, n_head, df_v)

        # Send output to input dim
        weighted_V = weighted_V.reshape((weighted_V.size(0), weighted_V.size(1), self.dx)) # Reshape (bs, n, dx)

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1) # (bs, 1, dx)
        yx2 = self.y_x_mul(y).unsqueeze(1) # (bs, 1, dx)
        newX_intermediate = yx1 + (yx2 + 1) * weighted_V # (bs, n, dx)

        # Output X
        newX = self.x_out(newX_intermediate) * x_mask # (bs, n, dx)
        # ---- MODIFICATION START ----
        if 'diffusion_utils' in globals() and hasattr(diffusion_utils, 'assert_correctly_masked'):
            diffusion_utils.assert_correctly_masked(newX, x_mask)
        # ---- MODIFICATION END ----

        # Process y based on X and E
        # Original y_y(y) was here. Let's assume y_y is an initial update to y.
        y_updated_by_self = self.y_y(y)
        e_y_contrib = self.e_y(E) # Use original E for y update, not newE
        x_y_contrib = self.x_y(X) # Use original X for y update, not newX
        
        new_y = y_updated_by_self + x_y_contrib + e_y_contrib
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.Module, act_fn_out: nn.Module, # Changed type hint for act_fn
                 # ---- MODIFICATION START ----
                 output_intermediate_y_at_layer: int = None, # Default to None
                 # ---- MODIFICATION END ----
                 addition: bool =True): # Added type hint for act_fn
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.addition = addition
        # ---- MODIFICATION START ----
        self.output_intermediate_y_at_layer = output_intermediate_y_at_layer
        if self.output_intermediate_y_at_layer is not None:
            if not (0 < self.output_intermediate_y_at_layer <= self.n_layers):
                raise ValueError(f"output_intermediate_y_at_layer must be between 1 and n_layers ({self.n_layers}), or None.")
        # ---- MODIFICATION END ----

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            dim_ffy=hidden_dims.get('dim_ffy', hidden_dims['dim_ffX'])) # Added dim_ffy with fallback
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n, device=X.device) # Added device
        diag_mask = ~diag_mask.bool() # Simplified
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, 1) # expand for broadcasting, ensure last dim is 1 for features

        X_to_out = X[..., :self.out_dim_X] if self.out_dim_X > 0 else None # Handle cases where out_dim can be 0
        E_to_out = E[..., :self.out_dim_E] if self.out_dim_E > 0 else None
        y_to_out = y[..., :self.out_dim_y] if self.out_dim_y > 0 else None


        new_E_after_mlp_in = self.mlp_in_E(E)
        # Symmetrize E after MLP input projection if it's meaningful
        # This was in original code, implies that edge features should be symmetric after this step
        new_E_after_mlp_in = (new_E_after_mlp_in + new_E_after_mlp_in.transpose(1, 2)) / 2
        
        # ---- MODIFICATION START ----
        # Ensuring utils.PlaceHolder is accessible.
        if 'utils' not in globals() or not hasattr(utils, 'PlaceHolder'):
            # This is a fallback if the import somehow failed or utils isn't structured as expected.
            # In a normal run, utils.PlaceHolder should be available.
            # For robustness, one might define a simple PlaceHolder locally or ensure the import path.
            class TempPlaceHolder: # Minimal placeholder for structure
                def __init__(self, X, E, y): self.X, self.E, self.y = X, E, y
                def mask(self, node_mask):
                    x_mask = node_mask.unsqueeze(-1)
                    self.X = self.X * x_mask
                    e_mask1 = x_mask.unsqueeze(2)
                    e_mask2 = x_mask.unsqueeze(1)
                    self.E = self.E * e_mask1 * e_mask2
                    return self
            _PlaceHolder = TempPlaceHolder
        else:
            _PlaceHolder = utils.PlaceHolder

        after_in = _PlaceHolder(X=self.mlp_in_X(X), E=new_E_after_mlp_in, y=self.mlp_in_y(y)).mask(node_mask)
        current_X, current_E, current_y = after_in.X, after_in.E, after_in.y

        intermediate_y_output = None

        for i, layer in enumerate(self.tf_layers):
            current_X, current_E, current_y = layer(current_X, current_E, current_y, node_mask)
            if self.output_intermediate_y_at_layer is not None and \
               (i + 1) == self.output_intermediate_y_at_layer: # (i+1) because layers are 1-indexed for user
                intermediate_y_output = current_y.clone()
        
        X_after_tf = current_X
        E_after_tf = current_E
        y_after_tf = current_y
        # ---- MODIFICATION END ----

        X_out_mlp = self.mlp_out_X(X_after_tf)
        E_out_mlp = self.mlp_out_E(E_after_tf)
        y_out_mlp = self.mlp_out_y(y_after_tf)

        if self.addition:
            if X_to_out is not None and self.out_dim_X > 0 : X_final = X_out_mlp + X_to_out 
            else: X_final = X_out_mlp
            
            if E_to_out is not None and self.out_dim_E > 0 : E_final = E_out_mlp + E_to_out
            else: E_final = E_out_mlp
            
            if y_to_out is not None and self.out_dim_y > 0 : y_final = y_out_mlp + y_to_out
            else: y_final = y_out_mlp
        else:
            X_final, E_final, y_final = X_out_mlp, E_out_mlp, y_out_mlp


        E_final = E_final * diag_mask # Apply diag_mask to the final E
        E_final = 1/2 * (E_final + torch.transpose(E_final, 1, 2)) # Symmetrize final E

        final_output_placeholder = _PlaceHolder(X=X_final, E=E_final, y=y_final).mask(node_mask)

        # ---- MODIFICATION START ----
        if self.output_intermediate_y_at_layer is not None:
            return final_output_placeholder, intermediate_y_output
        else:
            # To maintain consistency for unpacking, always return two values.
            # If not returning intermediate_y, return None for the second value.
            return final_output_placeholder, None
        # ---- MODIFICATION END ----