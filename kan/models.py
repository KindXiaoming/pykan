from kan import *
from kan.MLP import MLP

# define the AutoEncoder class

class AutoEncoder(KAN):
    
    def __init__(self, width_enc=None, width_dec=None, grid=3, k=3, seed=1, enc_type='kan', dec_type='kan'):
        
        # this is a bit hacky. The class is inherited from the KAN class to make it easy to create the fit() method.
        super(AutoEncoder, self).__init__(width=[1,1])
        
        if enc_type == 'kan':
            self.encoder = KAN(width=width_enc, grid=grid, k=k, seed=seed, auto_save=False, base_fun='identity')
        elif enc_type == 'mlp':
            self.encoder = MLP(width=width_enc, seed=seed)
            
        if dec_type == 'kan':
            self.decoder = KAN(width=width_dec, grid=grid, k=k, seed=seed, auto_save=False, base_fun='identity')
        elif dec_type == 'mlp':
            self.decoder = MLP(width=width_dec, seed=seed)
        
        self.enc_type = enc_type
        self.dec_type = dec_type
        
    def forward(self, x, singularity_avoiding=False, y_th=1000.):
        hidden = self.encoder(x)
        y = self.decoder(hidden)
        return y
    
    def get_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
    
    def get_reg(self, reg_metric='edge_backward', lamb_l1=1., lamb_entropy=2., lamb_coef=1., lamb_coefdiff=0.):
        
        if self.enc_type == 'kan':
            enc_reg = self.encoder.reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
        else:
            enc_reg = self.encoder.reg(reg_metric='w', lamb_l1=lamb_l1, lamb_entropy=lamb_entropy)
        
        if self.dec_type == 'kan':
            dec_reg = self.decoder.reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
        else:
            dec_reg = self.decoder.reg(reg_metric='w', lamb_l1=lamb_l1, lamb_entropy=lamb_entropy)
            
        return enc_reg + dec_reg
    
    def attribute(self):
        self.decoder.attribute()
        self.encoder.attribute()
            
        
    def update_grid(self, x):
        
        if self.enc_type == 'kan':
            self.encoder.update_grid_from_samples(x)
            
        if self.dec_type == 'kan':
            hidden = self.encoder(x)
            self.decoder.update_grid_from_samples(hidden)
            
    def disable_symbolic_in_fit(self):
        #self.disable_symbolic_in_fit()
        enc_symbolic_enabled = None
        dec_symbolic_enabled = None
        
        if self.enc_type == 'kan':
            enc_symbolic_enabled = self.encoder.symbolic_enabled
            self.encoder.disable_symbolic_in_fit()
       
        if self.dec_type == 'kan':
            dec_symbolic_enabled = self.decoder.symbolic_enabled
            self.decoder.disable_symbolic_in_fit()
            
        return enc_symbolic_enabled, dec_symbolic_enabled
    
    def disable_save_act_in_fit(self, lamb):
        
        old_save_act = self.save_act
        
        if lamb == 0.:
            self.save_act = False
        
        if self.enc_type == 'kan':
            self.encoder.disable_save_act_in_fit(lamb)
            
        if self.dec_type == 'kan':
            self.decoder.disable_save_act_in_fit(lamb)
            
        return old_save_act
            
    def recover_symbolic_in_fit(self, old_symbolic_enabled):
        self.symbolic_enabled = old_symbolic_enabled
        if self.enc_type == 'kan':
            self.encoder.symbolic_enabled = old_symbolic_enabled[0]
        if self.dec_type == 'kan':
            self.decoder.symbolic_enabled = old_symbolic_enabled[1]
        
    def recover_save_act_in_fit(self):
        self.save_act = True
        if self.enc_type == 'kan':
            self.encoder.save_act = True
        if self.dec_type == 'kan':
            self.decoder.save_act = True
        
            
    def fit(self, dataset, reg_metric='edge_backward', steps=20, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=1., lamb_coefdiff=0.):
        super(AutoEncoder, self).fit(dataset, reg_metric=reg_metric, steps=steps, lamb=lamb, lamb_l1=lamb_l1, lamb_entropy=lamb_entropy, lamb_coef=lamb_coef, lamb_coefdiff=lamb_coefdiff)
        