from model import resolve_single
from model.srgan import generator
from imresize import imresize
import matplotlib.pyplot as plt
import imageio
from utils import load_image, plot_sample
import time

from model.edsr import edsr


#model = edsr(scale=4, num_res_blocks=16)
#model.load_weights('weights/edsr-16-x4/weights.h5')



model = generator()
model.load_weights('weights/srgan/gan_generator.h5')
start_time = time.time()
original = load_image('demo/thermal_1m.png')

#the lines under need to be run when the image has 4 channels(1 extra transperacy layer).
#print(original.shape)
lr = original[...,:3]
#print(original.shape)

#lr = imresize(original,0.0625, method='bicubic')
#imageio.imwrite('thermal_downscaled.png', lr)
#lr = load_image('demo/db0b7bbca77296c6216b1d108a0e90e3.jpg')
sr = resolve_single(model, lr)
print("--- %s seconds ---" % (time.time() - start_time))
#sr_x2 = resolve_single(model, sr)
#sr_x3 = resolve_single(model, sr_x2)

imageio.imwrite('result/srganx1_32x32.png', sr)
#imageio.imwrite('result/edsrx2_32x32.png', sr_x2)
#imageio.imwrite('result/edsrx3_32x32.png', sr_x3)
#plot_sample(lr, sr, sr_x2)
#plt.savefig("out4.png")
#plt.show()