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
import numpy as np
import torch.nn.init
import random
# captum.attr
#from captum.attr import RISE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage.transform import resize 
from tqdm import tqdm 
from PIL import Image
import os
#from rise import RISE #EXTRA
#from utils.visualize import visualize, reverse_normalize #EXTRA

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=10, type=int, #Orignial son 100
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=200, type=int, #Cambié a 100, regresar a 1000
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.01, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.input_dim = input_dim
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

# load image
im = cv2.imread(args.input)
data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:
    mask = cv2.imread(args.input.replace('.'+args.input.split('.')[-1],'_scribble.png'),-1)
    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
    inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )
    inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )
    target_scr = torch.from_numpy( mask.astype(np.int) )
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable( target_scr )
    # set minLabels
    args.minLabels = len(mask_inds)

# train
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))

for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

    outputHP = output.reshape( (im.shape[0], im.shape[1], args.nChannel) )
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy,HPy_target)
    lhpz = loss_hpz(HPz,HPz_target)

    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        cv2.imshow( "output", im_target_rgb )
        cv2.waitKey(10)

    # loss 
    if args.scribble:
        loss = args.stepsize_sim * loss_fn(output[ inds_sim ], target[ inds_sim ]) + args.stepsize_scr * loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) + args.stepsize_con * (lhpy + lhpz)
    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
        
    loss.backward()
    optimizer.step()

    print (batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    im_target_rgb = cv2.resize(im_target_rgb,(224,224))
cv2.imwrite("output_45.jpg", im_target_rgb)

torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available()


image_path = './BSD500/hem4.bmp'
transform = transforms.Compose([
    transforms.Resize((30, 30)),#REDIMENSIONA LA IMAGEN
    transforms.ToTensor(), # CONVIERTE A TENSOR
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])#NORMALIZA CON MEDIA Y DESVESTA
])
image = Image.open(image_path)#ABRE LA IMAGEN
input_image = transform(image).unsqueeze(0)#APLICA LA TRANSFORMACION Y AGREGA DIMENSION EXTRA
#EN LA PRIMERA POSICION DE LA IMAGEN
input_image = input_image.to('cuda')

#model = model.to('cuda')
# Step 2: Load the trained model
#model = resnet50(pretrained=True)
#model.eval()
model = model.to('cuda')

# Step 3: Generate random masks
num_masks = 7000 #CANTIDAD DE MASCARAS
mask_size = input_image.size()[2:]  #OBTENER DIMENSIONES DE ALTO Y ANCHO 
s = 10#TAMAÑO DE LA MASCARA, valor por default 8
#NO USAR VALORES ALTOS DE MÁSCARAS
p1 = 0.9 #PROBABILIDAD DE ESTABLECER UN PIXEL EN 1, valor por default 0.5
masks = torch.zeros((num_masks,) + mask_size) #CREA TENSOR DE CEROS (NUM_MASKS, ALTO, ANCHO)
masks = masks.to('cuda')
for i in range(num_masks): #ITERA CADA MÁSCARA EN EL RANGO NUM_MASKS
    random_mask = torch.randint(0, 2, mask_size).float() #SE GENERA MASCARA ALEATORIA CON VALOR  BINARIO ENTRE 0 Y 1
    for j in range(s):
        p = torch.rand(1).item()
        if p < p1: 
            random_mask[int(torch.randint(0, mask_size[0], (1,)).item())][int(torch.randint(0, mask_size[1], (1,)).item())] = 1.0
    masks[i] = random_mask #ASIGNA LA MÁSCARA ALEATORIA GENERADA AL TENSOR DE MÁSCARAS EN EL CONTADOR i
 
# Step 4: Apply masks to the image
masked_images = input_image * masks.unsqueeze(1) #MULTIPLICA ELEMENTO POR LEMENTO LA IMAGEN DE ENTRADA POR LAS MASCARAS
#UNSQUEEZE(1) AGREGA DIMENSION EXTRA AL TENSOR PARA QUE COINCIDA CON LA DIMENSION DE LOS CANALES DE ENTRADA


# Step 5: Pass masked images through the model
with torch.no_grad():#EVITA EL CALCULO DE LOS GRADIENTES 
    outputs = model(masked_images)#PASA LAS IAMGENES ENMASCARADAS ATRAVÉS DEL MODELO 

# Step 6: Calculate importance scores
original_output = model(input_image)#PASS LA IMAGEN ORIGNAL ATRAVÉS DEL MODELO 
#PARA OBTENER LA SALIDA ORIGINAL SIN MÁSCARAS

#del outputs, original_output  # Libera los tensores
importance_scores = torch.sum((outputs - original_output) ** 2, dim=1)#CALCULA EL PUNTAJE DE IMPORTANCIA PARA 
#CADA IMAGEN ENMASCARADA, RESULTADO UN TENSOR CON LOS PUNTAJES DE IMPORTANCIA 

# Step 7: Visualize the importance map
importance_map = torch.mean(importance_scores, dim=0) #MEDIA DE LOS VALORES DE IMPORTANCIA 
#PROPORCIONA UN SOLO MAPA DE IMPORTANCIA EN LUGAR DE VARIOS 
importance_map = (importance_map - torch.min(importance_map)) / (torch.max(importance_map) - torch.min(importance_map))
#NORMALIZA EL MAPA PARA QUE LOS VALORES ESTEN ENTRE 0 Y 1 
importance_map = importance_map.detach().cpu().numpy()#CONVIERTE EL TENSOR DE PYTORCH EN UN ARREGLO NUMPY 
importance_map = (importance_map * 255).astype(np.uint8) #ESCALA LOS VALORES DE 0 A 255 Y LUEGO A UINT8 PARA QUE SEA IMAGEN 

# Aplicar el mapa de colores "jet" utilizando Matplotlib
plt.imshow(importance_map, cmap='jet')
plt.axis('off')
plt.colorbar()

# Guardar la figura en un archivo temporal utilizando Matplotlib
temp_file = 'heatmap_temp.png'
plt.savefig(temp_file, bbox_inches='tight', pad_inches=0)

# Leer la imagen temporal con OpenCV
heatmap_image = cv2.imread(temp_file)

# Eliminar el archivo temporal
os.remove(temp_file)

# Guardar la imagen con el mapa de colores "jet" utilizando OpenCV
cv2.imwrite('heatmap_jet_45_0.9_10_7000_gpu_30x30_200iter_lr0.01(18).png', heatmap_image)
#########


