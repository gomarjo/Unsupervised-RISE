#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import random
import glob
import datetime
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--maxUpdate', metavar='T', default=1000, type=int, 
                    help='number of maximum update count')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--batch_size', metavar='bsz', default=1, type=int, 
                    help='number of batch_size')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FOLDERNAME',
                    help='input image folder name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=5, type=float, 
                    help='step size for continuity loss')
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

img_list = sorted(glob.glob(args.input+'/ref/*'))
im = cv2.imread(img_list[0])

# train
model = MyNet( im.shape[2] )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

for batch_idx in range(args.maxIter):
    print('Training started. '+str(datetime.datetime.now())+'   '+str(batch_idx+1)+' / '+str(args.maxIter))
    for im_file in range(int(len(img_list)/args.batch_size)):
        for loop in tqdm.tqdm(range(args.maxUpdate)):
            im = []
            for batch_count in range(args.batch_size):
                # load image
                resized_im = cv2.imread(img_list[args.batch_size*im_file + batch_count])
                resized_im = cv2.resize(resized_im, dsize=(224, 224))
                resized_im = resized_im.transpose( (2, 0, 1) ).astype('float32')/255.
                im.append(resized_im)

            data = torch.from_numpy( np.array(im) )
            if use_cuda:
                data = data.cuda()
            data = Variable(data)
    
            HPy_target = torch.zeros(data.shape[0], resized_im.shape[1]-1, resized_im.shape[2], args.nChannel)
            HPz_target = torch.zeros(data.shape[0], resized_im.shape[1], resized_im.shape[2]-1, args.nChannel)
            if use_cuda:
                HPy_target = HPy_target.cuda()
                HPz_target = HPz_target.cuda()

            # forwarding
            optimizer.zero_grad()
            output = model( data )
            output = output.permute( 0, 2, 3, 1 ).contiguous().view( data.shape[0], -1, args.nChannel )

            outputHP = output.reshape( (data.shape[0], resized_im.shape[1], resized_im.shape[2], args.nChannel) )
    
            HPy = outputHP[:, 1:, :, :] - outputHP[:, 0:-1, :, :]
            HPz = outputHP[:, :, 1:, :] - outputHP[:, :, 0:-1, :]    
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            output = output.reshape( output.shape[0] * output.shape[1], -1 )
            ignore, target = torch.max( output, 1 )

            loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(args.input, 'b'+str(args.batch_size)+'_itr'+str(args.maxIter)+'_layer'+str(args.nConv+1)+'.pth'))

label_colours = np.random.randint(255,size=(100,3))
test_img_list = sorted(glob.glob(args.input+'/test/*'))
if not os.path.exists(os.path.join(args.input, 'result/')):
    os.mkdir(os.path.join(args.input, 'result/'))
print('Testing '+str(len(test_img_list))+' images.')
for img_file in tqdm.tqdm(test_img_list):
    im = cv2.imread(img_file)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    inds = target.data.cpu().numpy().reshape( (im.shape[0], im.shape[1]) )
    inds_rgb = np.array([label_colours[ c % args.nChannel ] for c in inds])
    inds_rgb = inds_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite( os.path.join(args.input, 'result/') + os.path.basename(img_file), inds_rgb )



# Ruta de la carpeta que contiene las imágenes generadas
result_folder = './BBC_3/result'

# Obtener la lista de archivos de imagen en la carpeta
image_files = os.listdir(result_folder)

# Procesar cada imagen en la carpeta
for image_file in image_files:
    # Ruta completa de la imagen
    image_path = os.path.join(result_folder, image_file)

    # Cargar la imagen como un array NumPy
    image = np.array(Image.open(image_path))
    
    #Obtener las dimensiones 
    image_height, image_width, _ = image.shape

    # Convertir la imagen de NumPy a un tensor de PyTorch y aplicar transformaciones
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to('cuda')

    # Generar máscaras aleatorias (aquí generamos una sola para simplificar)
    mask_size = input_image.size()[2:]
    s = 10
    p1 = 0.5
    random_mask = torch.zeros(mask_size).to('cuda')
    for i in range(s):
        p = torch.rand(1).item()
        if p < p1:
            random_mask[int(torch.randint(0, mask_size[0], (1,)).item())][int(torch.randint(0, mask_size[1], (1,)).item())] = 1.0

    # Aplicar la máscara a la imagen de entrada
    masked_image = input_image * random_mask.unsqueeze(0)

    # Obtener la salida del modelo con la máscara
    with torch.no_grad():
        output = model(masked_image)

    # Calcular la importancia
    original_output = model(input_image)
    importance_scores = torch.sum((output - original_output) ** 2, dim=1)

    # Visualizar el mapa de importancia
    importance_map = torch.mean(importance_scores, dim=0)
    importance_map = (importance_map - torch.min(importance_map)) / (torch.max(importance_map) - torch.min(importance_map))
    importance_map = importance_map.detach().cpu().numpy()
    importance_map = (importance_map * 255).astype(np.uint8)

    # Aplicar el mapa de colores "jet" utilizando OpenCV
    heatmap_image = cv2.applyColorMap(importance_map, cv2.COLORMAP_JET)

    # Superponer el mapa de calor en la imagen original
    result_image = cv2.addWeighted(image, 0.7, heatmap_image, 0.3, 0)

    # Visualizar y guardar la imagen con el mapa de calor
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.colorbar()
    plt.show()

    # Guardar la imagen con el mapa de calor
    output_path = os.path.join(result_folder, 'heatmap_' + image_file)
    cv2.imwrite(output_path, result_image)
    
    
    
#Esta modificación debería permitirte cargar imágenes NumPy y aplicar las transformaciones de PyTorch correctamente.

#Con esta modificación, el código debería funcionar correctamente incluso si las imágenes están en formato PIL al principio.








