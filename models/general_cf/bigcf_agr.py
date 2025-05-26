import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
from config.configurator import configs
from models.aug_utils import AdaptiveMask, NodeMask
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

def kernel_matrix(x, sigma):
    """Calculate Gaussian kernel matrix for HSIC computation - compatible with LLM recommendation framework"""
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)

def hsic(Kx, Ky, m):
    """Calculate Hilbert-Schmidt Independence Criterion (HSIC) value - used to measure independence between two distributions"""
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class BIGCF_AGR(BaseModel):
    def __init__(self, data_handler):
        super(BIGCF_AGR, self).__init__(data_handler)

        # prepare adjacency matrix for base model
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)),
                                  shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        self.all_h_list = list(plain_adj.row)
        self.all_t_list = list(plain_adj.col)
        self.A_in_shape = plain_adj.shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor(
            [list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))],
            dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()
        
        # Enhanced adjacency matrix from LLM, providing additional collaborative relationship information
        self.aug_adj = data_handler.aug_torch_adj if hasattr(data_handler, 'aug_torch_adj') else None
        
        # Adaptive masking tool
        self.adaptive_masker = AdaptiveMask(head_list=self.all_h_list, tail_list=self.all_t_list,
                                            matrix_shape=self.A_in_shape)

        # Get dataset name for specific configuration
        self.dataset = configs['data']['name']
        
        # Initialize dataset-specific parameters
        self._init_dataset_config()

        # Model basic parameters
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        
        # Structural knowledge embeddings for bidirectional knowledge distillation
        self.user_str_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embedding_size)), requires_grad=True)
        self.item_str_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embedding_size)), requires_grad=True)

        # Training/testing state flags
        self.is_training = True
        self.final_embeds = None

        # Semantic embeddings from LLM
        self.usrprf_embeds = torch.tensor(configs['user_embedding']).float().cuda()
        self.itmprf_embeds = torch.tensor(configs['item_embedding']).float().cuda()
        
        # Merge semantic embeddings for reconstruction task
        self.prf_embeds = torch.concat([self.usrprf_embeds, self.itmprf_embeds], dim=0)
        
        # Node masking for self-supervised learning
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        
        # Knowledge distillation MLP, used to transfer LLM semantic knowledge to the recommendation model
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        
        # Generator network MLP, used to reconstruct semantic embeddings from collaborative filtering embeddings
        self.gen_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, (self.prf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.prf_embeds.shape[1] + self.embedding_size) // 2, self.prf_embeds.shape[1])
        )
        
        # Adaptive graph structure learning module
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=2*self.embedding_size, out_features=self.embedding_size, bias=True)
        self.linear_2 = nn.Linear(in_features=self.embedding_size, out_features=1, bias=True)
        
        # Edge weight adjustment parameters
        self.edge_bias = configs['model'].get('edge_bias', 0.5)
        self.delta = 0.01  # Coefficient controlling the influence of structural knowledge

        self._init_weight()

    def _init_dataset_config(self):
        """Initialize dataset-specific configuration parameters"""
        if self.dataset in configs['model']:
            dataset_config = configs['model'][self.dataset]
            # Basic hyperparameters
            self.layer_num = dataset_config.get('layer_num', configs['model']['layer_num'])
            self.reg_weight = dataset_config.get('reg_weight', configs['model']['reg_weight'])
            self.cl_weight = dataset_config.get('cl_weight', configs['model']['cl_weight'])
            self.cl_temperature = dataset_config.get('cl_temperature', configs['model']['cl_temperature'])
            self.kd_weight = dataset_config.get('kd_weight', configs['model']['kd_weight'])
            self.kd_temperature = dataset_config.get('kd_temperature', configs['model']['kd_temperature'])
            self.cen_weight = dataset_config.get('cen_weight', configs['model']['cen_weight'])
            self.intent_num = configs['model']['intent_num']
            
            # Additional LLM-AGR specific parameters
            self.prf_weight = dataset_config.get('prf_weight', configs['model'].get('prf_weight', 0.1))
            self.mask_ratio = dataset_config.get('mask_ratio', configs['model'].get('mask_ratio', 0.1))
            self.recon_weight = dataset_config.get('recon_weight', configs['model'].get('recon_weight', 0.1))
            self.re_temperature = dataset_config.get('re_temperature', configs['model'].get('re_temperature', 0.2))
            self.beta = dataset_config.get('beta', configs['model'].get('beta', 5.0))  # HSIC regularization coefficient
            self.sigma = dataset_config.get('sigma', configs['model'].get('sigma', 0.25))  # Gaussian kernel parameter
            self.str_weight = dataset_config.get('str_weight', configs['model'].get('str_weight', 1.0))  # Structural knowledge weight
            self.alpha = dataset_config.get('alpha', configs['model'].get('alpha', 0.1))  # LLM knowledge integration weight
        else:
            # Default configuration parameters
            self.layer_num = configs['model']['layer_num']
            self.reg_weight = configs['model']['reg_weight']
            self.cl_weight = configs['model']['cl_weight']
            self.cl_temperature = configs['model']['cl_temperature']
            self.kd_weight = configs['model']['kd_weight']
            self.kd_temperature = configs['model']['kd_temperature']
            self.cen_weight = configs['model']['cen_weight']
            self.intent_num = configs['model']['intent_num']
            
            # Additional LLM-AGR specific parameters
            self.prf_weight = configs['model'].get('prf_weight', 0.1)
            self.mask_ratio = configs['model'].get('mask_ratio', 0.1)
            self.recon_weight = configs['model'].get('recon_weight', 0.1)
            self.re_temperature = configs['model'].get('re_temperature', 0.2)
            self.beta = configs['model'].get('beta', 5.0)  # HSIC regularization coefficient
            self.sigma = configs['model'].get('sigma', 0.25)  # Gaussian kernel parameter
            self.str_weight = configs['model'].get('str_weight', 1.0)  # Structural knowledge weight
            self.alpha = configs['model'].get('alpha', 0.1)  # LLM knowledge integration weight

    def _init_weight(self):
        # Initialize basic embedding weights
        init(self.user_embedding.weight)
        init(self.item_embedding.weight)
        
        # Initialize knowledge distillation MLP weights
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
        
        # Initialize generator network MLP weights
        for m in self.gen_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
                
        # Initialize graph learner weights
        init(self.linear_1.weight)
        init(self.linear_2.weight)

    def _cal_sparse_adj(self):
        """Calculate normalized graph adjacency matrix"""
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values
    
    def _mask(self):
        """Node masking method for self-supervised learning tasks"""
        embeds = torch.concat([self.user_embedding.weight, self.item_embedding.weight], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds
    
    def learn_graph_structure(self, cf_index=None):
        """Adaptive graph structure learner, optimizing graph connection weights through neural networks"""
        if cf_index is None:
            return self.G_indices, self.G_values
            
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        head_list, tail_list = self.all_h_list, self.all_t_list
        row_emb = all_emb[head_list]
        col_emb = all_emb[tail_list]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        
        # Edge connection weight calculation
        out_layer1 = self.activate(self.linear_1(cat_emb))
        logit = self.linear_2(out_layer1).view(-1)
        
        # Gumbel-Softmax reparameterization trick, using fixed temperature 0.2
        eps = torch.rand(logit.shape).to(logit.device)
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        edge_weights = torch.sigmoid(mask_gate_input) + self.edge_bias
        
        # Create optimized denoising adjacency matrix values
        G_new_values = self.G_values * edge_weights
        
        return self.G_indices, G_new_values
    
    def _propagate(self, G_indices, G_values, embeddings):
        """Graph convolution propagation process"""
        return torch_sparse.spmm(G_indices, G_values, self.A_in_shape[0], self.A_in_shape[1], embeddings)

    def forward(self, G_indices=None, G_values=None, masked_user_embeds=None, masked_item_embeds=None, emb_type='cf'):
        """Model forward propagation, integrating LightGCN's multi-layer propagation and BIGCF's intent guidance"""
        if G_indices is None:
            G_indices = self.G_indices
        if G_values is None:
            G_values = self.G_values
            
        # Check cache for efficiency
        if not self.is_training and self.final_embeds is not None and emb_type == 'cf':
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None
        
        # Select initial embeddings based on embedding type and mask status
        if emb_type == 'cf':
            if masked_user_embeds is None or masked_item_embeds is None:
                user_init_embeds = self.user_embedding.weight
                item_init_embeds = self.item_embedding.weight
            else:
                user_init_embeds = masked_user_embeds
                item_init_embeds = masked_item_embeds
        elif emb_type == 'str':
            user_init_embeds = self.user_str_embeds
            item_init_embeds = self.item_str_embeds
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}")
        
        all_embeddings = [torch.concat([user_init_embeds, item_init_embeds], dim=0)]

        # Multi-layer graph convolutional propagation
        for i in range(0, self.layer_num):
            gnn_layer_embeddings = self._propagate(G_indices, G_values, all_embeddings[i])
            all_embeddings.append(gnn_layer_embeddings)

        # Aggregate multi-layer embeddings
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        # Bidirectional intent guidance (BIGCF feature)
        u_embeddings, i_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num], 0)
        u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
        i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T

        int_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

        # Reparameterization technique
        noise = torch.randn_like(all_embeddings)
        all_embeddings = all_embeddings + int_embeddings * noise

        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.user_num, self.item_num], 0)
        
        # Cache embeddings
        if emb_type == 'cf':
            self.final_embeds = all_embeddings

        return self.ua_embedding, self.ia_embedding, all_embeddings, int_embeddings

    def cal_cl_loss(self, users, items, gnn_emb, int_emb):
        """Calculate contrastive learning loss, retain BIGCF's featured functionality"""
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.cl_temperature)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.cl_temperature), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        u_gnn_embs, i_gnn_embs = torch.split(gnn_emb, [self.user_num, self.item_num], 0)
        u_int_embs, i_int_embs = torch.split(int_emb, [self.user_num, self.item_num], 0)

        u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
        u_int_embs = F.normalize(u_int_embs[users], dim=1)

        i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
        i_int_embs = F.normalize(i_int_embs[items], dim=1)

        cl_loss += cal_loss(u_gnn_embs, u_gnn_embs)
        cl_loss += cal_loss(i_gnn_embs, i_gnn_embs)
        cl_loss += cal_loss(u_gnn_embs, i_gnn_embs)

        cl_loss += cal_loss(u_int_embs, u_int_embs)
        cl_loss += cal_loss(i_int_embs, i_int_embs)
        return cl_loss

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        """Select corresponding embedding vectors from batch data"""
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    
    def _reconstruction(self, embeds, seeds):
        """Semantic information reconstruction task, mapping collaborative embeddings back to semantic space"""
        if seeds is None:
            return torch.tensor(0.0).cuda()
            
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.gen_mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss
    
    def hsic_graph(self, users, pos_items):
        """HSIC-based graph information bottleneck regularization, reducing redundant information"""
        # Calculate user embedding HSIC loss
        users = torch.unique(users)
        batch_size = users.shape[0]
        if batch_size < 2:  # Need at least 2 samples to calculate HSIC
            return torch.tensor(0.0).cuda()
            
        input_x = self.user_emb_old[users]
        input_y = self.ua_embedding[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, batch_size)
        
        # Calculate item embedding HSIC loss
        items = torch.unique(pos_items)
        batch_size = items.shape[0]
        if batch_size < 2:  # Need at least 2 samples to calculate HSIC
            return loss_user
            
        input_i = self.item_emb_old[items]
        input_j = self.ia_embedding[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, batch_size)
        
        loss = loss_user + loss_item
        return loss

    def cal_loss(self, batch_data):
        """Calculate overall model loss, integrating multiple learning objectives"""
        self.is_training = True
        self.batch_size = len(batch_data[0])  # Get batch size
        
        # 1. Node masking for self-supervised learning
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        
        # 2. Adaptive graph structure learning
        denoised_G_indices, denoised_G_values = self.learn_graph_structure(configs.get('cf_index', None))
        
        # 3. Calculate embeddings on original and optimized graphs
        self.user_emb_old, self.item_emb_old, old_gnn_embeddings, old_int_embeddings = self.forward(
            self.G_indices, self.G_values, masked_user_embeds, masked_item_embeds
        )
        
        self.ua_embedding, self.ia_embedding, gnn_embeddings, int_embeddings = self.forward(
            denoised_G_indices, denoised_G_values, masked_user_embeds, masked_item_embeds
        )
        
        # Get batch data
        ancs, poss, negs = batch_data
        
        # 4. Process prior knowledge provided by LLM
        # 4.1 Process semantic preference knowledge
        user_prf = self.mlp(self.usrprf_embeds)
        item_prf = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            user_prf, item_prf, batch_data
        )
        
        # 4.2 Process structural knowledge
        if self.aug_adj is not None:
            # Use enhanced adjacency matrix to calculate structural embeddings
            self.user_emb_str, self.item_emb_str, _, _ = self.forward(
                G_indices=None, G_values=None, emb_type='str'
            )
            ancstr_embeds, posstr_embeds, negstr_embeds = self._pick_embeds(
                self.user_emb_str, self.item_emb_str, batch_data
            )
        else:
            self.user_emb_str, self.item_emb_str = None, None
            ancstr_embeds, posstr_embeds, negstr_embeds = None, None, None
        
        # 5. Calculate BIGCF's main losses
        # ELBO reconstruction loss
        u_embeddings = self.ua_embedding[ancs]
        pos_embeddings = self.ia_embedding[poss]
        neg_embeddings = self.ia_embedding[negs]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # Regularization loss
        reg_loss = self.reg_weight * reg_params(self)
        
        # Intent centralization loss
        cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_weight * cen_loss
        
        # Contrastive learning loss
        cl_loss = self.cl_weight * self.cal_cl_loss(ancs, poss, gnn_embeddings, int_embeddings)
        
        # 6. Add LLM-AGR specific losses
        # 6.1 Preference knowledge contrastive learning loss
        prf_loss = cal_infonce_loss(u_embeddings, ancprf_embeds, user_prf, self.kd_temperature) + \
            cal_infonce_loss(pos_embeddings, posprf_embeds, item_prf, self.kd_temperature) + \
            cal_infonce_loss(neg_embeddings, negprf_embeds, item_prf, self.kd_temperature)
        prf_loss /= u_embeddings.shape[0]
        prf_loss *= self.prf_weight
        
        # 6.2 Structural knowledge contrastive learning loss
        if self.user_emb_str is not None:
            str_cl_loss = cal_infonce_loss(u_embeddings, ancstr_embeds, self.user_emb_str, self.kd_temperature) + \
                cal_infonce_loss(pos_embeddings, posstr_embeds, self.item_emb_str, self.kd_temperature) + \
                cal_infonce_loss(neg_embeddings, negstr_embeds, self.item_emb_str, self.kd_temperature)
            str_cl_loss /= u_embeddings.shape[0]
            str_loss = str_cl_loss * self.delta
            str_loss *= self.str_weight
        else:
            str_loss = torch.tensor(0.0).cuda()
            
        # 6.3 Reconstruction loss
        recon_loss = self._reconstruction(gnn_embeddings, seeds)
        recon_loss *= self.recon_weight
        
        # 6.4 Information bottleneck regularization
        ib_loss = self.hsic_graph(ancs, poss) * self.beta
        
        # 7. Total loss calculation
        loss_cf = mf_loss + reg_loss + cl_loss + cen_loss
        loss_llm = (prf_loss + str_loss + recon_loss) * self.alpha
        loss = loss_cf + loss_llm + ib_loss
        
        # Record loss components for analysis
        losses = {
            'mf_loss': mf_loss, 
            'reg_loss': reg_loss, 
            'cl_loss': cl_loss,
            'cen_loss': cen_loss,
            'prf_loss': prf_loss,
            'str_loss': str_loss,
            'recon_loss': recon_loss,
            'ib_loss': ib_loss
        }
        
        return loss, losses

    def full_predict(self, batch_data):
        """Generate complete recommendation predictions for model evaluation"""
        # Apply adaptive graph structure learning
        denoised_G_indices, denoised_G_values = self.learn_graph_structure(configs.get('cf_index', None))
        user_embeds, item_embeds, _, _ = self.forward(denoised_G_indices, denoised_G_values)
            
        self.is_training = False
        
        # Get users to recommend and training set mask
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        
        # Calculate user-item interaction scores
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        
        # Apply training set mask to avoid recommending already interacted items
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds