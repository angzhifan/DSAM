import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


def process_network_line(line, global_drop):
    # break line on the ; each segment is a parameter for the layer of that line. OK
    if  len(line)==0 or line[0]=='#':
        return None
    sss = str.split(line, ';')
    lp = {}
    for ss in sss:
        # Split between parameter name and value
        s = str.split(ss, ':')
        #s1 = str.strip(s[1], ' \n')
        # Process the parameter value
        # A nonlinearity function
        a = ''
        # A number
        s1 = str.strip(s[1], ' \n')
        try:
            a = float(s1)
            if '.' not in s1:
                a = int(s[1])
        # A tuple, a list or the original string
        except ValueError:
            if '(' in s[1]:
                aa = str.split(str.strip(s1, ' ()\n'), ',')
                a = []
                try:
                    int(aa[0])
                    for aaa in aa:
                        a.append(int(aaa))
                    a = tuple(a)
                except ValueError:
                    for aaa in aa:
                        a.append(float(aaa))
                    a = tuple(a)
            elif '[' in s[1]:
                aa = str.split(str.strip(s1, ' []\n'), ',')
                a = []
                for aaa in aa:
                    a.append(aaa)
            elif (s1 == 'None'):
                a = None
            elif (s1 == 'True'):
                a = True
            elif (s1 == 'False'):
                a = False
            else:
                a = s1
        # Add a global drop value to drop layers
        s0 = str.strip(s[0], ' ')
        if (s0 == 'drop' and global_drop is not None):
            lp[s0] = global_drop
        else:
            lp[s0] = a
    return (lp)

def get_network(layers,nf=None):


    LP=[]
    for line in layers:
        lp = process_network_line(line, None)
        if lp is not None:
            LP += [lp]
    layers_dict=LP
    if (nf is not None):
        LP[0]['num_filters']=nf

    return layers_dict


class temp_args(nn.Module):
    def __init__(self):
        super(temp_args, self).__init__()
        self.back = None
        self.first = 0
        self.everything = False
        self.layer_text = None
        self.dv = None
        self.optimizer = None
        self.embedd_layer = None
        KEYS = None


class Subsample(nn.Module):
    def __init__(self,stride=None):
        super(Subsample,self).__init__()

        self.stride=stride
        if self.stride is not None:
            if stride % 2 ==0:
                self.pd=0
            else:
                self.pd=(stride-1)//2


    def forward(self,z,dv):

        if self.stride is None:
            return(z)
        else:
            if self.pd>0:
                temp=torch.zeros(z.shape[0],z.shape[1],z.shape[2]+2*self.pd,z.shape[3]+2*self.pd).to(dv)
                temp[:,:,self.pd:self.pd+z.shape[2],self.pd:self.pd+z.shape[3]]=z
                tempss=temp[:,:,::self.stride,::self.stride]
            else:
                tempss=z[:,:,::self.stride,::self.stride]


        return(tempss)

class Inject(nn.Module):
    def __init__(self, ll,sh):
        super(Inject,self).__init__()

        self.ps=ll['stride']
        self.sh=sh
        self.feats=ll['num_filters']
        self.pad=ll['pad']
        self.pad2 = self.pad // 2

    def forward(self,input):

        if input.is_cuda:
            num=input.get_device()
            dv=torch.device('cuda:'+str(num))
        else:
            dv=torch.device('cpu')
        input=input.reshape(-1,self.feats, self.sh[0],self.sh[1])
        out=torch.zeros(input.shape[0],input.shape[1],input.shape[2]*self.ps+self.pad,input.shape[3]*self.ps+self.pad).to(dv)

        out[:,:,self.pad2:(self.pad2+input.shape[2]*self.ps):self.ps,(self.pad2):(self.pad2+input.shape[3]*self.ps):self.ps]=input

        return out

class CUT(nn.Module):
    def __init__(self, sh):
        super(CUT, self).__init__()
        self.sh=sh

    def forward(self,input):
        assert self.sh[0]<input.shape[2] and self.sh[1]< input.shape[3], (self.sh,input.shape)
        return(input[:,:,0:self.sh[0], 0:self.sh[1]])

class NONLIN(nn.Module):
    def __init__(self, type,low=-1., high=1.):
        super(NONLIN, self).__init__()
        self.type=type
        if 'HardT' in self.type:
            self.HT=nn.Hardtanh(low,high)
        if 'smx' in self.type:
            self.tau=high

    def forward(self,input):

        if ('HardT' in self.type):
            return(self.HT(input))
        elif ('tanh' in self.type):
            return(F.tanh(input))
        elif ('sigmoid' in self.type):
            return(torch.sigmoid(input))
        elif ('leaky' in self.type):
            return(F.leaky_relu(input))
        elif ('relu' in self.type):
            return(F.relu(input))
        elif ('smx' in self.type):
            return F.softmax(input*self.tau,dim=1)

        elif ('iden'):
            return(input)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()



    def forward(self,input,atemp=None,lay=None):

        #print('IN',input.shape, input.get_device())
        if atemp is None:
            atemp=self.temp

        everything = False

            #print('INP_dim',input.shape[0])
        in_dims={}
        if (atemp.first):
            self.layers = nn.ModuleList()

        OUTS={}
        old_name=''

        DONE=False
        for i,ll in enumerate(atemp.layer_text):
            if not DONE:
                inp_ind = old_name

                if ('parent' in ll):
                    pp=ll['parent']
                    # over ride default inp_feats
                    if len(pp)==1:
                        inp_ind=pp[0]
                        if atemp.first:
                            inp_feats=OUTS[pp[0]].shape[1]
                            in_dim=in_dims[pp[0]]
                    else:
                        inp_feats=[]
                        loc_in_dims=[]
                        inp_ind=[]
                        for p in pp:
                            inp_ind += [p]
                            if atemp.first:
                                inp_feats+=[OUTS[p].shape[1]]
                                loc_in_dims+=[in_dims[p]]
                if ('input' in ll['name']):
                    out=input
                    if atemp.first:
                        if 'shape' in ll and 'num_filters' in ll:
                            atemp.input_shape=[ll['num_filters']]+list(ll['shape'])

                     #   out=out.reshape(out.shape[0],ll['num_filters'],)
                    if everything:
                        OUTS[ll['name']]=out

                if ('conv' in ll['name']):
                    if everything:
                        out = OUTS[inp_ind]
                    # Reshape to grid based data with inp_feats features.
                    if len(out.shape)==2:
                        wdim=np.int(np.sqrt(out.shape[1]/inp_feats))
                        out=out.reshape(out.shape[0],inp_feats,wdim,wdim)
                    if atemp.first:
                        bis = True
                        if ('nb' in ll):
                            bis = False
                        stride=1;
                        if 'stride' in ll:
                            stride=ll['stride']
                        if 'pad' not in ll:
                            pd=(ll['filter_size']//stride) // 2
                        else:
                            pd=ll['pad']
                        self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,padding=pd, bias=bis))


                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if 'non_linearity' in ll['name']:
                    if atemp.first:
                        low=-1.; high=1.
                        if 'lims' in ll:
                            low=ll['lims'][0]; high=ll['lims'][1]
                        self.layers.add_module(ll['name'],NONLIN(ll['type'],low=low,high=high))
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out
                if ('Avg' in ll['name']):
                    if atemp.first:
                        HW=(np.int32(OUTS[inp_ind].shape[2]/2),np.int32(OUTS[inp_ind].shape[3]/2))
                        self.layers.add_module(ll['name'],nn.AvgPool2d(HW,HW))
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('pool' in ll['name']):
                    if atemp.first:
                        stride = ll['pool_size']
                        pp = (ll['pool_size'] - 1) // 2
                        if ('stride' in ll):
                            stride = ll['stride']
                            pp=1
                        #pp=[np.int32(np.mod(ll['pool_size'],2))]

                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], stride=stride, padding=pp))


                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('drop' in ll['name']):
                    if atemp.first:
                        self.layers.add_module(ll['name'],torch.nn.Dropout(p=ll['drop'], inplace=False))


                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('dense' in ll['name']):
                    if atemp.first:
                        out_dim=ll['num_units']
                        bis=True
                        if ('nb' in ll):
                            bis=False
                        self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis))

                    if everything:
                        out=OUTS[inp_ind]

                    out = out.reshape(out.shape[0], -1)
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out
                if 'cut' in ll['name']:
                    if atemp.first:
                        self.layers.add_module(ll['name'],CUT(ll['shape']))
                    if everything:
                        out = OUTS[inp_ind]
                    out = getattr(self.layers, ll['name'])(out)
                if 'inject' in ll['name']:
                    if atemp.first:
                        if 'shape' in ll:
                            sh=ll['shape']
                        else:
                            sh=prev_shape[2:4]
                        self.layers.add_module(ll['name'],Inject(ll,sh))
                    if everything:
                        out = OUTS[inp_ind]
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if 'subsample' in ll['name']:
                    if atemp.first:
                        stride = None
                        if 'stride' in ll:
                            stride = ll['stride']
                        self.layers.add_module(ll['name'], Subsample(stride=stride))

                    if everything:
                        out = OUTS[inp_ind]
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('norm') in ll['name']:
                    if atemp.first:
                        if atemp.bn=='full':
                            if len(prev_shape)==4 and atemp.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(prev_shape[1]))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(prev_shape[1]))
                        elif atemp.bn=='half_full':
                            if len(prev_shape)==4 and atemp.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(prev_shape[1], affine=False))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(prev_shape.shape[1], affine=False))
                        elif atemp.bn=='layerwise':
                                self.layers.add_module(ll['name'],nn.LayerNorm(OUTS[old_name].shape[2:4]))
                        elif atemp.bn=='instance':
                            self.layers.add_module(ll['name'], nn.InstanceNorm2d(OUTS[old_name].shape[1],affine=True))


                    if not atemp.first:
                        out = getattr(self.layers, ll['name'])(out)
                        if everything:
                            OUTS[ll['name']] = out
                    else:
                        pass


                if ('opr' in ll['name']):
                    if 'add' in ll['name']:
                        out = OUTS[inp_ind[0]]+OUTS[inp_ind[1]]
                        OUTS[ll['name']] = out
                        inp_feats=out.shape[1]
                if ('num_filters' in ll):
                    inp_feats = ll['num_filters']

                prev_shape=out.shape
                if atemp.first:
                    print(ll['name']+' '+str(np.array(prev_shape))+'\n')

                in_dim=np.prod(prev_shape[1:])
                in_dims[ll['name']]=in_dim
                old_name=ll['name']
                if lay is not None and lay in ll['name']:
                    DONE=True


        return out

def initialize_model(args, sh, layers,device, layers_dict=None):

    model=network()

    if layers_dict==None:
            layers_dict=get_network(layers)


    for l in layers_dict:
        # Over ride the nunber of units given in the arg file to equal the valuein sh[0]
        if 'dense_gaus' in l['name']:
            if sh is not None:
                l['num_units']=sh[0]

    atemp = temp_args()
    atemp.layer_text = layers_dict
    atemp.dv = device
    atemp.everything = False
    atemp.bn=args.bn


    if sh is not None:
        temp = torch.zeros([1] + list(sh))  # .to(device)
        # Run the network once on dummy data to get the correct dimensions.
        atemp.first=1
        atemp.input_shape=None
        bb = model.forward(temp,atemp)

        atemp.output_shape = bb.shape
        atemp.input_shape = sh

        atemp.first=0

        model.add_module('temp',atemp)

        model=model.to(atemp.dv)

        return model


class decoder_mix(nn.Module):
    def __init__(self,dv, args):
        super(decoder_mix,self).__init__()

        self.z_dim=args.latent_space_dimension
        self.u_dim=args.trans_space_dimension
        if self.u_dim>0:
            self.dec_trans_top=nn.ModuleList([initialize_model(args, (self.u_dim,), args.dec_trans_top, dv) for i in range(args.num_class)])

        self.dec_conv_top=nn.ModuleList([initialize_model(args, (self.z_dim,),args.dec_layers_top, dv)for i in range(args.num_class) ])
        f_shape=np.array(self.dec_conv_top[0].temp.output_shape)[1:]
        self.dec_conv_bot=initialize_model(args, f_shape, args.dec_layers_bot, dv)

        if hasattr(self.dec_conv_bot.layers, 'inject'):
           self.in_feats=self.dec_conv_bot.layers.inject.feats
           self.in_shape=self.dec_conv_bot.layers.inject.sh

    def forward(self, inp, cl):

        u=None
        if self.u_dim > 0:
            u = inp.narrow(len(inp.shape) - 1, 0, self.u_dim)
            z = inp.narrow(len(inp.shape) - 1, self.u_dim, self.z_dim)
            pu = self.dec_trans_top[cl](u)[0]
        else:
            z = inp

        x = self.dec_conv_top[cl](z)
        x = self.dec_conv_bot(x)

        x = torch.sigmoid(x)
        x = x.reshape(x.shape[0],-1)
        return x, u

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu