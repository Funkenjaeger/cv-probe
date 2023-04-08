from __future__ import print_function
import argparse
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

distance = 4
image1 = cv2.imread('img\cvtest_'+str(distance)+'_0.png')
image2 = cv2.imread('img\cvtest_'+str(distance)+'_0.25.png')
image3 = cv2.imread('img\cvtest_'+str(distance)+'_1.png')
#image = cv2.imread(args["image1"])

#image = image[:, 790:810]
#print("width: {} pixels".format(image.shape[1]))
#print("height: {} pixels".format(image.shape[0]))
#print("channels: {}".format(image.shape[2]))

#i1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#i2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
i1 = np.sum(image1.astype('float'), axis=2)[:, 800:820]
i2 = np.sum(image2.astype('float'), axis=2)[:, 800:820]
i3 = np.sum(image3.astype('float'), axis=2)[:, 800:820]

i1 -= np.mean(i1)
i2 -= np.mean(i2)
i3 -= np.mean(i3)

#result = cv2.matchTemplate(i1, i2, cv2.TM_CCOEFF)

xcorr = scipy.signal.fftconvolve(i1, i1[::-1, ::-1], 'same')

fig = plt.figure()
ax = plt.axes(projection='3d')
z = scipy.signal.fftconvolve(i1, i1[::-1, ::-1], 'same')
x = np.linspace(0, 1, z.shape[0])
y = np.linspace(0, 1, z.shape[1])
xx, yy = np.meshgrid(x, y)
ax.plot_surface(xx, yy, np.transpose(scipy.signal.fftconvolve(i1, i1[::-1, ::-1], 'same')), edgecolor='none')
ax.plot_surface(xx, yy, np.transpose(scipy.signal.fftconvolve(i1, i2[::-1, ::-1], 'same')), edgecolor='none')
ax.plot_surface(xx, yy, np.transpose(scipy.signal.fftconvolve(i1, i3[::-1, ::-1], 'same')), edgecolor='none')

corr1 = np.max(scipy.signal.fftconvolve(i1, i1[::-1, ::-1], 'same'), 1)
corr2 = np.max(scipy.signal.fftconvolve(i1, i2[::-1, ::-1], 'same'), 1)
corr3 = np.max(scipy.signal.fftconvolve(i1, i3[::-1, ::-1], 'same'), 1)

print('Autocorrelation peak at offset ' + str(np.argmax(corr1)-600))
print('0 x 0.25" xcorr peak at offset ' + str(np.argmax(corr2)-600))
print('0 x 1" xcorr peak at offset ' + str(np.argmax(corr3)-600))
plt.figure(1)
plt.plot(corr1, label='1x1')
plt.plot(corr2, label='1x2')
plt.plot(corr3, label='1x3')
plt.title('xcorr')
plt.legend()

plt.figure(2)
plt.plot(np.diff(corr1), label='1x1')
plt.plot(np.diff(corr2), label='1x2')
plt.plot(np.diff(corr3), label='1x3')
plt.title('xcorr derivative')

plt.figure(3)
plt.plot(np.abs(np.diff(np.diff(corr1))), label='1x1')
plt.plot(np.abs(np.diff(np.diff(corr2))), label='1x2')
plt.plot(np.abs(np.diff(np.diff(corr3))), label='1x3')
plt.title('|xcorr 2nd derivative|')
plt.legend()

plt.show()