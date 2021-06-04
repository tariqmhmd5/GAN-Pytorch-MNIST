import PIL
import torch
import random
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

digit = 3
    
n0 = 200

def to_list(img):
    return list(map(int, img.view((28*28,)).tolist()))
    
SCALE_OFF = 0    
SCALE_RANGE = 1
SCALE_01 = 2

def show_image(tens, imgname=None, scale=SCALE_01):
    """
    Show an image contained in a tensor. The tensor will be reshaped properly, as long as it has the required 28*28 = 784 entries.
    
    If imgname is provided, the image will be saved to a file, otherwise it will be stored in a temporary file and displayed on screen.
    
    The parameter scale can be used to perform one of three scaling operations:
        SCALE_OFF: No scaling is performed, the data is expected to use values between 0 and 255
        SCALE_RANGE: The data will be rescaled from whichever scale it has to be between 0 and 255. This is useful for data in an unknown/arbitrary range. The lowest value present in the data will be 
        converted to 0, the highest to 255, and all intermediate values will be assigned using linear interpolation
        SCALE_01: The data will be rescaled from a range between 0 and 1 to the range between 0 and 255. This can be useful if you normalize your data into that range.
    """
    r = tens.max() - tens.min()
    img = PIL.Image.new("L", (28,28))
    scaled = tens
    if scale == SCALE_RANGE:
        scaled = (tens - tens.min())*255/r
    elif scale == SCALE_01:
        scaled = tens*255
    img.putdata(to_list(scaled))
    if imgname is None:
        img.show()
    else:
        img.save(imgname)
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784,512)
        self.hddn = nn.Linear(512,256)
        self.hddn1 = nn.Linear(256,128)
        self.out = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.hddn(x))
        x = F.leaky_relu(self.hddn1(x))
        x = torch.sigmoid(self.out(x))
        return x

def train_classifier(opt, model, x, y):
    criterion = nn.BCELoss()
    for i in range(n0):
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred,y)
        loss.backward()
        print(f'Epoch: {i}, Loss: {loss.item():.4f}')
        opt.step()
    return model


def classify(x_train, y_train, x_validation, labels_validation):
    network = Discriminator()
    opt = torch.optim.Adam(network.parameters(), lr=0.0002)
    network = train_classifier(opt,network,x_train,y_train)
    
    with torch.no_grad():
        p = network(x_validation)
    
    y_pred = (p > 0.5).float()
    y_val_true = labels_validation
    
    tp = (y_pred * y_val_true).sum()
    fp = (y_pred * (1-y_val_true)).sum()
    fn = ((1-y_pred) * y_val_true).sum()
    tn = ((1-y_pred) * (1-y_val_true)).sum()
    
    a = y_pred.sum()/x_validation.shape[0] *100
    b = tp/labels_validation.shape[0] *100
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    
    print(f'percentage of images that were of that digit were predicted as chosen digit: {a}% \n percentage of digits that were classified correctly: {b}% \n Accuracy: {accuracy:.4f}% \t Precision: {precision}\t Recall: {recall}')
    
    
# (GAN) starts here

# Change number of total training iterations for GAN, for the discriminator and for the generator
n = 8
n1 = 8
n2 = 8

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(100, 128)
        self.hdn = nn.Linear(128,256)
        self.hdn1 = nn.Linear(256,512)
        self.out = nn.Linear(512,784)
        
    def forward(self, x):
        x = F.leaky_relu(self.inp(x))
        x = F.leaky_relu(self.hdn(x))
        x = F.leaky_relu(self.hdn1(x))
        x = torch.tanh(self.out(x))
        return x

def train_discriminator(opt, discriminator, x_true, x_false):
    print("Training discriminator")
    criterion = nn.BCELoss()
    for i in range(n1):
        opt.zero_grad()
        pred = discriminator(x_true)
        loss_true = criterion(pred,torch.ones(x_true.shape[0],1))
        loss_true.backward()
        
        pred_false = discriminator(x_false)
        loss_false = criterion(pred_false,torch.zeros(x_false.shape[0],1))
        loss_false.backward()
        
        print(f'Epoch: {i}, Loss_true: {loss_true.item():.4f}, Loss_False: {loss_false.item():.4f}')
        opt.step()
        
def train_generator(opt, generator, discriminator):
    print("Training generator")
    criterion = nn.BCELoss()
    for i in range(n2):
        opt.zero_grad()
        noice = torch.rand((100,100)).detach()
        g_pred = generator(noice)
        d_pred = discriminator(g_pred)
        loss = criterion(d_pred,torch.ones(noice.shape[0],1))
        loss.backward()
        print(f'Epoch: {i}, Loss_D: {loss.item():.4f}')
        opt.step()

def gan(x_real):
    show_image(x_real[0], "train_0.png", scale=SCALE_01)
    x_false = torch.rand((100,784)).detach()
    noice = torch.rand((100,100)).detach()
    d = Discriminator()
    g = Generator()
    opt_d = torch.optim.Adam(d.parameters(), lr=0.0002)
    opt_g = torch.optim.Adam(g.parameters(), lr=0.0002)
    for i in range(n):
        train_discriminator(opt_d, d, x_real,x_false)
        train_generator(opt_g, g, d)
    with torch.no_grad():
        gen_images = g(noice)           
    for j,(img,fake) in enumerate(zip(gen_images,x_false)):
        img = img.reshape(28,28)
        fake = fake.reshape(28,28)
        save_image(img, 'gen_images/generated/gen'+str(j)+'.png') #saving fake images
        save_image(fake, 'gen_images/Fake/fake'+str(j)+'.png')   #saving generated images

def main(rungan):
    """
        automatically download the data set if it doesn't exist yet
        make sure all tensor shapes are correct
        normalize the images (all pixels between 0 and 1)
        provide labels for the classification task (0 for all images that are not your digit, 1 for the ones that are)
        extract the images of your chosen digit for the GAN
    """
    train = torchvision.datasets.MNIST(".", download=True)
    x_train = train.data.float().view(-1,28*28)/255.0
    labels_train = train.targets
    y_train = (labels_train == digit).float().view(-1,1)
    
    validation = torchvision.datasets.MNIST(".", train=False)
    x_validation = validation.data.float().view(-1,28*28)/255.0
    labels_validation = validation.targets
    labels_validation = (labels_validation == digit).float().view(-1,1)
    
    if rungan:
        gan(x_train[labels_train == digit])
    else:
        classify(x_train, y_train, x_validation, labels_validation)
        
"""
You can pass -g or --gan to the script to run the GAN part, otherwise it will run the classification part.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a classifier or a GAN on the MNIST data.')
    parser.add_argument('--gan', '-g', dest='gan', action='store_const',
                        const=True, default=False,
                        help='Train and run the GAN (default: train and run the classifier only)')

    args = parser.parse_args()
    main(args.gan)





