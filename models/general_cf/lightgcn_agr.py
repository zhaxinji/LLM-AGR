import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.aug_utils import NodeMask
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, ssl_con_loss
from models.base_model import BaseModel
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

class LightGCN_AGR(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_AGR, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        
        # Enhanced adjacency matrix from LLM, providing additional collaborative relationship information
        self.aug_adj = data_handler.aug_torch_adj if hasattr(data_handler, 'aug_torch_adj') else self.adj
        
        # Get configuration parameters
        self.keep_rate = configs['model']['keep_rate']
        self.edge_bias = configs['model'].get('edge_bias', 0.5)
        
        # Basic embedding vectors for users and items
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        # Structural embedding vectors for users and items, learned from the enhanced adjacency matrix
        self.user_str_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_str_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # Get dataset-specific hyperparameter configuration
        self.dataset = configs['data']['name']
        self._init_dataset_config()
        
        # User and item semantic embeddings from LLM
        self.usrprf_embeds = t.tensor(configs['user_embedding']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['item_embedding']).float().cuda()
        
        # Merge semantic embeddings for reconstruction task
        self.prf_embeds = t.concat([self.usrprf_embeds, self.itmprf_embeds], dim=0)
        
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
        self.delta = 0.01  # Coefficient controlling the influence of structural knowledge

        self._init_weight()

    def _init_dataset_config(self):
        """Initialize dataset-specific configuration parameters"""
        if self.dataset in configs['model']:
            dataset_config = configs['model'][self.dataset]
            # Basic hyperparameters
            self.layer_num = dataset_config.get('layer_num', configs['model']['layer_num'])
            self.reg_weight = dataset_config.get('reg_weight', configs['model']['reg_weight'])
            self.prf_weight = dataset_config.get('prf_weight', configs['model']['prf_weight'])
            self.kd_temperature = dataset_config.get('kd_temperature', configs['model']['kd_temperature'])
            
            # Mask reconstruction parameters
            self.mask_ratio = dataset_config.get('mask_ratio', configs['model']['mask_ratio'])
            self.recon_weight = dataset_config.get('recon_weight', configs['model']['recon_weight'])
            self.re_temperature = dataset_config.get('re_temperature', configs['model']['re_temperature'])
            
            # Graph regularization parameters
            self.beta = dataset_config.get('beta', configs['model'].get('beta', 5.0))  # HSIC regularization coefficient
            self.sigma = dataset_config.get('sigma', configs['model'].get('sigma', 0.25))  # Gaussian kernel parameter
            self.str_weight = dataset_config.get('str_weight', configs['model'].get('str_weight', 1.0))  # Structural knowledge weight
            self.alpha = dataset_config.get('alpha', configs['model'].get('alpha', 0.1))  # LLM knowledge integration weight
        else:
            # Default configuration parameters
            self.layer_num = configs['model']['layer_num']
            self.reg_weight = configs['model']['reg_weight']
            self.prf_weight = configs['model']['prf_weight']
            self.kd_temperature = configs['model']['kd_temperature']
            self.mask_ratio = configs['model']['mask_ratio']
            self.recon_weight = configs['model']['recon_weight']
            self.re_temperature = configs['model']['re_temperature']
            self.beta = configs['model'].get('beta', 5.0)  # HSIC regularization coefficient
            self.sigma = configs['model'].get('sigma', 0.25)  # Gaussian kernel parameter
            self.str_weight = configs['model'].get('str_weight', 1.0)  # Structural knowledge weight
            self.alpha = configs['model'].get('alpha', 0.1)  # LLM knowledge integration weight

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
    
    def _propagate(self, adj, embeds):
        """Graph convolution information propagation process"""
        return t.spmm(adj, embeds)
    
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
    
    def forward(self, adj=None, keep_rate=1.0, emb_type='cf', masked_user_embeds=None, masked_item_embeds=None):
        """Forward propagation calculation, generating user and item embedding representations"""
        if adj is None:
            adj = self.adj
            
        # Check cache for efficiency
        if not self.is_training and self.final_embeds is not None and emb_type == 'cf':
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
            adj = self.edge_dropper(adj, keep_rate)
            
        # Multi-layer graph convolutional network propagation
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
            
        # Use LightGCN's simple layer summation strategy
        embeds = sum(embeds_list)
            
        # Cache final embeddings to speed up inference
        if emb_type == 'cf':
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
        """Calculate overall model loss, integrating multiple learning objectives"""
        self.is_training = True
        self.batch_size = len(batch_data[0])  # Get batch size
        
        # 1. Node masking for self-supervised learning
        masked_user_embeds, masked_item_embeds, seeds = self._mask()
        
        # 2. Adaptive graph structure learning
        denoised_adj = self.learn_graph_structure(self.adj, configs.get('cf_index', None))
        
        # 3. Calculate embeddings on original and optimized graphs
        self.user_emb_old, self.item_emb_old = self.forward(
            self.adj, self.keep_rate, 'cf', masked_user_embeds, masked_item_embeds
        )
        
        self.user_emb, self.item_emb = self.forward(
            denoised_adj, self.keep_rate, 'cf', masked_user_embeds, masked_item_embeds
        )
        
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(
            self.user_emb, self.item_emb, batch_data
        )
        
        # 4. Process prior knowledge provided by LLM
        # 4.1 Process semantic preference knowledge
        user_prf = self.mlp(self.usrprf_embeds)
        item_prf = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            user_prf, item_prf, batch_data
        )
        
        # 4.2 Process structural knowledge
        self.user_emb_str, self.item_emb_str = self.forward(self.aug_adj, self.keep_rate, 'str')
        ancstr_embeds, posstr_embeds, negstr_embeds = self._pick_embeds(
            self.user_emb_str, self.item_emb_str, batch_data
        )
        
        # 5. Calculate collaborative filtering main loss
        auc, bpr_loss, reg_loss = self._bpr_loss(batch_data)
        
        llm_auc, llm_bpr_loss, llm_reg_loss = self._bpr_loss(batch_data, 'str')
        
        # Preference knowledge contrastive learning loss
        prf_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, user_prf, self.kd_temperature) + \
            cal_infonce_loss(pos_embeds, posprf_embeds, item_prf, self.kd_temperature) + \
            cal_infonce_loss(neg_embeds, negprf_embeds, item_prf, self.kd_temperature)
        prf_loss /= anc_embeds.shape[0]
        prf_loss *= self.prf_weight
        
        # Structural knowledge contrastive learning loss
        str_cl_loss = cal_infonce_loss(anc_embeds, ancstr_embeds, self.user_emb_str, self.kd_temperature) + \
            cal_infonce_loss(pos_embeds, posstr_embeds, self.item_emb_str, self.kd_temperature) + \
            cal_infonce_loss(neg_embeds, negstr_embeds, self.item_emb_str, self.kd_temperature)
        str_bpr_loss = (llm_bpr_loss + llm_reg_loss)
        str_cl_loss /= anc_embeds.shape[0]
        str_loss = str_bpr_loss + str_cl_loss * self.delta
        str_loss *= self.str_weight
        
        # 6. Information bottleneck regularization, reducing redundant information
        users, pos_items, neg_items = batch_data
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        
        # 7. Total loss calculation
        loss_cf = bpr_loss + reg_loss
        loss_llm = (prf_loss + str_loss) * self.alpha
        loss = loss_cf + loss_llm + ib_loss
        
        # Record loss components for analysis
        losses = {
            'bpr_loss': bpr_loss, 
            'reg_loss': reg_loss, 
            'prf_loss': prf_loss,
            'str_loss': str_loss,
            'ib_loss': ib_loss
        }
        
        return loss, losses
    
    def _bpr_loss(self, batch_data, g_type='cf'):
        """Bayesian Personalized Ranking loss calculation, for different types of embeddings"""
        users, pos_items, neg_items = batch_data
        
        if g_type == 'cf':
            users_emb = self.user_emb[users]
            pos_emb = self.item_emb[pos_items]
            neg_emb = self.item_emb[neg_items]
            users_emb_ego = self.user_embeds[users]
            pos_emb_ego = self.item_embeds[pos_items]
            neg_emb_ego = self.item_embeds[neg_items]
        elif g_type == 'str':
            users_emb = self.user_emb_str[users]
            pos_emb = self.item_emb_str[pos_items]
            neg_emb = self.item_emb_str[neg_items]
            users_emb_ego = self.user_str_embeds[users]
            pos_emb_ego = self.item_str_embeds[pos_items]
            neg_emb_ego = self.item_str_embeds[neg_items]
        else:
            return 0, t.tensor(0.0).cuda(), t.tensor(0.0).cuda()
        
        reg_loss = 1/2 * (users_emb_ego.norm(2).pow(2) +
                  pos_emb_ego.norm(2).pow(2) +
                  neg_emb_ego.norm(2).pow(2)) / float(len(users))
                  
        pos_scores = t.sum(t.mul(users_emb, pos_emb), dim=1)
        neg_scores = t.sum(t.mul(users_emb, neg_emb), dim=1)
        
        auc = t.mean((pos_scores > neg_scores).float())
        bpr_loss = t.mean(-t.log(t.sigmoid(pos_scores - neg_scores) + 1e-9))
        
        return auc, bpr_loss, reg_loss * self.reg_weight

    def full_predict(self, batch_data):
        """Generate complete recommendation predictions for model evaluation"""
        # Apply adaptive graph structure learning
        denoised_adj = self.learn_graph_structure(self.adj, configs.get('cf_index', None))
        user_embeds, item_embeds = self.forward(denoised_adj, 1.0)
            
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