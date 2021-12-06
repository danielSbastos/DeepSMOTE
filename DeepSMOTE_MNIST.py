import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import pandas as pd

print(torch.version.cuda) #10.1
t3 = time.time()
##############################################################################
"""args for AE"""

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 1#3    # number of channels in the input data 

args['n_z'] = 300 #600     # number of dimensions in latent space. 

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 2       # how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'mnist'  #'fmnist' # specify which dataset to use


##############################################################################



## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(8, 4, 4),
            nn.LeakyReLU(0.2, inplace=True),
        )#,
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.fc = nn.Sequential(
            nn.Linear(1, 4),
            nn.LeakyReLU(0.2, inplace=True))

        self.deconv = nn.Sequential(
            nn.Conv1d(10, 80, 1),
            nn.ReLU(True),
            #nn.Sigmoid())
            )
            # nn.Tanh())

    def forward(self, x):
        x = self.fc(x)

        x = x.reshape(1, 10, 4)

        import pdb; pdb.set_trace()
        x = self.deconv(x)
        return x
#NOTE: Download the training ('.../0_trn_img.txt') and label files 
# ('.../0_trn_lab.txt').  Place the files in directories (e.g., ../MNIST/trn_img/
# and /MNIST/trn_lab/).  Originally, when the code was written, it was for 5 fold
#cross validation and hence there were 5 files in each of the 
#directories.  Here, for illustration, we use only 1 training and 1 label
#file (e.g., '.../0_trn_img.txt' and '.../0_trn_lab.txt').

# dtrnimg = './MNIST/trn_img/'
# dtrnlab = './MNIST/trn_lab/'

# ids = os.listdir(dtrnimg)
# idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
# print(idtri_f)

# ids = os.listdir(dtrnlab)
# idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]
# print(idtrl_f)

dtrn_iris = './MNIST/trn_iris/'
ids = os.listdir(dtrn_iris)
idtr_iris_f = 'iris.csv'

#for i in range(5):
for i in range(len(ids)):
    print()
    print(i)
    encoder = Encoder(args)
    decoder = Decoder(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    train_on_gpu = torch.cuda.is_available()

    #decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    df = pd.read_csv('./MNIST/trn_iris/iris-unb.csv')
    df = df.drop('Unnamed: 0', axis=1)
    dec_x = df.drop('variety', axis=1)
    # dex_x = torch.from_numpy(dec_x)
    dec_y = df['variety']

    batch_size = 100
    num_workers = 0

    #torch.Tensor returns float so if want long then use torch.tensor
    dec_x_r = np.reshape
    tensor_x = torch.Tensor(np.array(dec_x)).reshape(10, -1, 4)

    tensor_y = torch.Tensor(np.array(dec_y)).reshape(10, -1, 1)

    mnist_bal = TensorDataset(tensor_x,tensor_y)
    train_loader = torch.utils.data.DataLoader(mnist_bal, 
        batch_size=batch_size,shuffle=True,num_workers=num_workers)

    best_loss = np.inf

    t0 = time.time()
    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr = args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr = args['lr'])
    
        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0
            # train for one epoch -- set nets to train mode
            encoder.train()
            decoder.train()
        
            for images,labs in train_loader:
            
                # zero gradients for each batch
                encoder.zero_grad()
                decoder.zero_grad()
                images, labs = images.to(device), labs.to(device)
                #print('images ',images.size()) 
                labsn = labs.detach().cpu().numpy()
                #print('labsn ',labsn.shape, labsn)
                # run images
                z_hat = encoder(images)
                x_hat = decoder(z_hat) #decoder outputs tanh
                #print('xhat ', x_hat.size())
                #print(x_hat)
                mse = criterion(x_hat,images)
                #print('mse ',mse)
                resx = []
                resy = []
            
                tc = np.random.choice(10,1)
                tc = 2
                xbeg = dec_x[dec_y == tc]
                ybeg = dec_y[dec_y == tc] 
                xlen = len(xbeg)
                nsamp = min(xlen, 100)
                ind = np.random.choice(list(range(len(xbeg))),nsamp,replace=False)
                xclass = xbeg[ind]
                yclass = ybeg[ind]
            
                xclen = len(xclass)
                #print('xclen ',xclen)
                xcminus = np.arange(1,xclen)
                #print('minus ',xcminus.shape,xcminus)
                
                xcplus = np.append(xcminus,0)
                #print('xcplus ',xcplus)
                xcnew = (xclass[[xcplus],:])
                #xcnew = np.squeeze(xcnew)
                xcnew = xcnew.reshape(xcnew.shape[1],xcnew.shape[2],xcnew.shape[3],xcnew.shape[4])
                #print('xcnew ',xcnew.shape)
            
                xcnew = torch.Tensor(xcnew)
                xcnew = xcnew.to(device)
            
                #encode xclass to feature space
                xclass = torch.Tensor(xclass)
                xclass = xclass.to(device)
                xclass = encoder(xclass)
                #print('xclass ',xclass.shape) 
            
                xclass = xclass.detach().cpu().numpy()
            
                xc_enc = (xclass[[xcplus],:])
                xc_enc = np.squeeze(xc_enc)
                #print('xc enc ',xc_enc.shape)
            
                xc_enc = torch.Tensor(xc_enc)
                xc_enc = xc_enc.to(device)
                
                ximg = decoder(xc_enc)
                
                mse2 = criterion(ximg,xcnew)
            
                comb_loss = mse2 + mse
                comb_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
            
                train_loss += comb_loss.item()*images.size(0)
                tmse_loss += mse.item()*images.size(0)
                tdiscr_loss += mse2.item()*images.size(0)
            
                 
            # print avg training statistics 
            train_loss = train_loss/len(train_loader)
            tmse_loss = tmse_loss/len(train_loader)
            tdiscr_loss = tdiscr_loss/len(train_loader)
            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                    train_loss,tmse_loss,tdiscr_loss))
            
        
        
            #store the best encoder and decoder models
            #here, /crs5 is a reference to 5 way cross validation, but is not
            #necessary for illustration purposes
            if train_loss < best_loss:
                print('Saving..')
                path_enc = './MNIST/models/crs5/' \
                    + str(i) + '/bst_enc.pth'
                path_dec = './MNIST/models/crs5/' \
                    + str(i) + '/bst_dec.pth'
             
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
        
                best_loss = train_loss
        
        
        #in addition, store the final model (may not be the best) for
        #informational purposes
        path_enc = '.../MNIST/models/crs5/' \
            + str(i) + '/f_enc.pth'
        path_dec = '.../MNIST/models/crs5/' \
            + str(i) + '/f_dec.pth'
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
        print()
              
    t1 = time.time()
    print('total time(min): {:.2f}'.format((t1 - t0)/60))             
 
t4 = time.time()
print('final time(min): {:.2f}'.format((t4 - t3)/60))

