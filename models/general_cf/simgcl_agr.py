import pickle
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.aug_utils import NodeMask
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

def kernel_matrix(x, sigma):
    """Calculate Gaussian kernel matrix for HSIC computation - compatible with LLM recommendation framework"""
    return t.exp((t.matmul(x, x.transpose(0,1)) - 1) / sigma)

def hsic(Kx, Ky, m):
    """Calculate Hilbert-Schmidt Independence Criterion (HSIC) value - used to measure independence between two distributions"""
    Kxy = t.mm(Kx, Ky)
    h = t.trace(Kxy) / m ** 2 + t.mean(Kx) * t.mean(Ky) - \
        2 * t.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

class SimGCL_AGR(LightGCN):
    def __init__(self, data_handler):
        super(SimGCL_AGR, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        
        # Enhanced adjacency matrix from LLM, providing additional collaborative relationship information
        self.aug_adj = data_handler.aug_torch_adj if hasattr(data_handler, 'aug_torch_adj') else self.adj
        
        # Get configuration parameters
        self.keep_rate = configs['model']['keep_rate']
        self.edge_bias = configs['model'].get('edge_bias', 0.5)
        
        # Retain original SimGCL_plus parameters
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight'] 
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.eps = self.hyper_config['eps']
        
        # Add AGR-specific parameters
        self.dataset = configs['data']['name']
        self._init_dataset_config()
        
        # Structural embedding vectors for users and items, learned from the enhanced adjacency matrix
        self.user_str_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_str_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False
        
        # User and item semantic embeddings from LLM
        self.usrprf_embeds = t.tensor(configs['user_embedding']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['item_embedding']).float().cuda()
        
        # Merge semantic embeddings for reconstruction task
        self.prf_embeds = t.concat([self.usrprf_embeds, self.itmprf_embeds], dim=0)
        
        # Node masking for self-supervised learning
        self.masker = NodeMask(self.mask_ratio, self.embedding_size)
        
        # Knowledge distillation MLP, retained from SimGCL_plus
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )
        
        # Generator network MLP, used to reconstruct semantic embeddings from collaborative filtering embeddings (bidirectional knowledge distillation)
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
        self.delta = 0.01  # Coefficient controlling the influence of structural knowledge

        self._init_weight()
        
    def _init_dataset_config(self):
        """Initialize dataset-specific configuration parameters"""
        if self.dataset in configs['model']:
            dataset_config = configs['model'][self.dataset]
            # Integrate AGR-specific hyperparameters, retain SimGCL_plus original parameters
            self.prf_weight = dataset_config.get('prf_weight', configs['model'].get('prf_weight', 1.0e-2))
            self.mask_ratio = dataset_config.get('mask_ratio', configs['model'].get('mask_ratio', 0.2))
            self.recon_weight = dataset_config.get('recon_weight', configs['model'].get('recon_weight', 1.0e-2))
            self.re_temperature = dataset_config.get('re_temperature', configs['model'].get('re_temperature', 0.2))
            self.beta = dataset_config.get('beta', configs['model'].get('beta', 5.0))  # HSIC regularization coefficient
            self.sigma = dataset_config.get('sigma', configs['model'].get('sigma', 0.25))  # Gaussian kernel parameter
            self.str_weight = dataset_config.get('str_weight', configs['model'].get('str_weight', 1.0))  # Structural knowledge weight
            self.alpha = dataset_config.get('alpha', configs['model'].get('alpha', 0.1))  # LLM knowledge integration weight
        else:
            # Default configuration parameters
            self.prf_weight = configs['model'].get('prf_weight', 1.0e-2)
            self.mask_ratio = configs['model'].get('mask_ratio', 0.2)
            self.recon_weight = configs['model'].get('recon_weight', 1.0e-2)
            self.re_temperature = configs['model'].get('re_temperature', 0.2)
            self.beta = configs['model'].get('beta', 5.0)
            self.sigma = configs['model'].get('sigma', 0.25)
            self.str_weight = configs['model'].get('str_weight', 1.0)
            self.alpha = configs['model'].get('alpha', 0.1)

    def _init_weight(self):
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

    def _perturb_embedding(self, embeds):
        """Embedding perturbation function retained from SimGCL_plus"""
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise
    
    def _mask(self):
        """Node masking method for self-supervised learning tasks"""
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        masked_embeds, seeds = self.masker(embeds)
        return masked_embeds[:self.user_num], masked_embeds[self.user_num:], seeds
    
    def learn_graph_structure(self, adj=None, cf_index=None):
        """Adaptive graph structure learner, optimizing graph connection weights through neural networks"""
        if adj is None or cf_index is None:
            return self.adj
            
        all_emb = t.cat([self.user_embeds, self.item_embeds], dim=0)
        row, col = cf_index[:, 0], cf_index[:, 1]
        row_emb = all_emb[row]
        col_emb = all_emb[col]
        cat_emb = t.cat([row_emb, col_emb], dim=1)
        
        # Edge connection weight calculation
        out_layer1 = self.activate(self.linear_1(cat_emb))
        logit = self.linear_2(out_layer1).view(-1)
        
        # Gumbel-Softmax reparameterization trick, using fixed temperature 0.2
        eps = t.rand(logit.shape).to(logit.device)
        mask_gate_input = t.log(eps) - t.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        mask_gate_input = t.sigmoid(mask_gate_input) + self.edge_bias
        
        # Create optimized denoising adjacency matrix
        masked_Graph = t.sparse.FloatTensor(
            adj.indices(), 
            adj.values() * mask_gate_input, 
            t.Size([self.user_num + self.item_num, self.user_num + self.item_num])
        )
        
        return masked_Graph.coalesce()
    
    def forward(self, adj=None, perturb=False, emb_type='cf', masked_user_embeds=None, masked_item_embeds=None):
        """Forward propagation calculation, integrating SimGCL perturbation and AGR functionality"""
        if adj is None:
            adj = self.adj
            
        # Check cache for efficiency
        if not self.is_training and self.final_embeds is not None and emb_type == 'cf' and not perturb:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        
        # Select initial embeddings based on embedding type and mask status
        if emb_type == 'cf':
            if masked_user_embeds is None or masked_item_embeds is None:
                embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
            else:
                embeds = t.concat([masked_user_embeds, masked_item_embeds], axis=0)
        elif emb_type == 'str':
            embeds = t.concat([self.user_str_embeds, self.item_str_embeds], axis=0)
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}")
            
        embeds_list = [embeds]
        
        # Edge dropping during training to enhance model robustness
        if self.is_training:
            adj = self.edge_dropper(adj, self.keep_rate)
            
        # Multi-layer graph convolutional network propagation
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            # Perturb embeddings (from SimGCL) if needed
            if perturb:
                embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
            
        # Use LightGCN's layer summation strategy
        embeds = sum(embeds_list)
            
        # Cache final embeddings to speed up inference
        if emb_type == 'cf' and not perturb:
            self.final_embeds = embeds
            
        return embeds[:self.user_num], embeds[self.user_num:]

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
            return t.tensor(0.0).cuda()
            
        enc_embeds = embeds[seeds]
        prf_embeds = self.prf_embeds[seeds]
        enc_embeds = self.gen_mlp(enc_embeds)
        recon_loss = ssl_con_loss(enc_embeds, prf_embeds, self.re_temperature)
        return recon_loss
    
    def hsic_graph(self, users, pos_items):
        """HSIC-based graph information bottleneck regularization, reducing redundant information"""
        # Calculate user embedding HSIC loss
        users = t.unique(users)
        input_x = self.user_emb_old[users]
        input_y = self.user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        
        # Calculate item embedding HSIC loss
        items = t.unique(pos_items)
        input_i = self.item_emb_old[items]
        input_j = self.item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, self.batch_size)
        
        loss = loss_user + loss_item
        return loss

    def cal_loss(self, batch_data):
        """Calculate overall model loss, integrating SimGCL and AGR loss components"""
        self.is_training = True
        self.batch_size = len(batch_data[0])  # Get batch size
        
        # 1. Node masking for self-supervised learning
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        
        # 2. Adaptive graph structure learning
        denoised_adj = self.learn_graph_structure(self.adj, configs.get('cf_index', None))
        
        # 3. Calculate SimGCL perturbed embeddings (CL view 1 and view 2)
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
        
        # 4. Calculate embeddings on original and optimized graphs
        self.user_emb_old, self.item_emb_old = self.forward(
            self.adj, perturb=False, emb_type='cf', masked_user_embeds=masked_user_embeds, 
            masked_item_embeds=masked_item_embeds
        )
        
        self.user_emb, self.item_emb = self.forward(
            denoised_adj, perturb=False, emb_type='cf', masked_user_embeds=masked_user_embeds, 
            masked_item_embeds=masked_item_embeds
        )
        
        # 5. Get batch data embedding vectors
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            self.user_emb, self.item_emb, batch_data
        )
        
        # 6. Process SimGCL perturbed embeddings, calculate contrastive loss
        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight
        
        # 7. Process prior knowledge provided by LLM
        # 7.1 Process semantic preference knowledge
        user_prf = self.mlp(self.usrprf_embeds)
        item_prf = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            user_prf, item_prf, batch_data
        )
        
        # Reconstruction loss - mapping from collaborative filtering to semantic space
        recon_loss = self._reconstruction(
            t.concat([self.user_emb, self.item_emb], dim=0), seeds
        ) * self.recon_weight
        
        # 7.2 Process structural knowledge
        self.user_emb_str, self.item_emb_str = self.forward(self.aug_adj, perturb=False, emb_type='str')
        ancstr_embeds, posstr_embeds, negstr_embeds = self._pick_embeds(
            self.user_emb_str, self.item_emb_str, batch_data
        )
        
        # 8. Calculate collaborative filtering main loss
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        
        # Unidirectional knowledge distillation loss (from LLM to CF)
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, user_prf, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, item_prf, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, item_prf, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight
        
        # Structural knowledge distillation loss
        str_cl_loss = cal_infonce_loss(anc_embeds, ancstr_embeds, self.user_emb_str, self.kd_temperature) + \
                     cal_infonce_loss(pos_embeds, posstr_embeds, self.item_emb_str, self.kd_temperature) + \
                     cal_infonce_loss(neg_embeds, negstr_embeds, self.item_emb_str, self.kd_temperature)
        str_cl_loss /= anc_embeds.shape[0]
        str_loss = str_cl_loss * self.delta * self.str_weight
        
        # 9. Information bottleneck regularization, reducing redundant information
        users, pos_items, neg_items = batch_data
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        
        # 10. Total loss calculation, integrating all learning objectives
        loss_cf = bpr_loss + reg_loss + cl_loss  # Original SimGCL loss
        loss_llm = (kd_loss + str_loss + recon_loss) * self.alpha  # AGR-extended LLM-related loss
        loss = loss_cf + loss_llm + ib_loss
        
        # Record loss components for analysis
        losses = {
            'bpr_loss': bpr_loss, 
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'str_loss': str_loss,
            'recon_loss': recon_loss,
            'ib_loss': ib_loss
        }
        
        return loss, losses

    def full_predict(self, batch_data):
        """Generate complete recommendation predictions for model evaluation"""
        # Apply adaptive graph structure learning
        denoised_adj = self.learn_graph_structure(self.adj, configs.get('cf_index', None))
        user_embeds, item_embeds = self.forward(denoised_adj, perturb=False)
            
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