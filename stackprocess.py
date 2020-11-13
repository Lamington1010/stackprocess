"""
@author: Lamington1010
Copyright 2020 by Lamington1010
All rights reserved.
This file is part of the stackprocess tool
and is released under the "MIT License Agreement". Please see the LICENSE
file that should have been included as part of this package.

"""

#import all the necessary libraries 
import cv2
import numpy
import matplotlib.pyplot as plt

#define variables
circareatot = 0
rectareatot = 0
minrectareatot = 0
contareatot = 0 
concavetot = 0
convextot = 0

#define empty arrays
areacirc = []
arearec = []
areaminrec = []
areacont = []
concavem = []
convexm = []
thresholdmatrix = []
centerx = []
centery = []

#define the font of what is to be displayed next to the objects
font = cv2.FONT_HERSHEY_COMPLEX

for q in range(1,18):
    #read the image
    """ Add file location here """
    img = cv2.imread()
    
    img = img[224:1824, 224:1824]
    imagematrix.append(img)
    #large 3D matrix of all images
    
    
    # global thresholding
    ret1,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,40,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering

    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #maks countours for just th3
    _,contours, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0),1)
        #print(len(approx))
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        
        #locate the centers of each centroid
        M = cv2.moments(cnt)
        
        #object centers
        cX = int((M["m10"]+1e-5) / (M["m00"] + 1e-5))
        cY = int((M["m01"]+1e-5) / (M["m00"]+ 1e-5))
        centerx.append(cX)
        centery.append(cY)
        
        
        #concaveorconvex
        k = cv2.isContourConvex(cnt)
        if k == True:
            convextot = convextot + 1
        if k == False:
            concavetot = concavetot + 1
        
        #contour area
        contarea = cv2.contourArea(cnt)
        
        #box that binds contoured objects
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w, y+h), (255,0,0),1)
        
    
        #generate min area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        cv2.drawContours(img,[box],0,(255,0,0))
    
        #min area enclosing circle 
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (255, 0, 0), 1)
        
        #area calculations
        rectarea = int(h)*int(w)
        circarea = 3.14*radius*radius
        
        #calculate min rectangle area
        w1 = (((box[2,0]-box[1,0])**2)+((box[2,1]-box[1,1])**2))**0.5
        l1 = (((box[0,0]-box[1,0])**2)+((box[0,1]-box[1,1])**2))**0.5
        d1 = (((box[0,0]-box[2,0])**2)+((box[0,1]-box[2,1])**2))**0.5
        s1 = (w1+l1+d1)/2
        minrectarea = 2*((s1*(s1-w1)*(s1-l1)*(s1-d1)))**0.5
        
        
        #display values for each object
        print("Contour: ", int(contarea), "Circle: ", int(circarea), " Rect: ", int(rectarea), "MinRect: ", int(minrectarea))
        
        #display location of each object in image
        cv2.putText(img, "Loc: " + str(round(cX)) + " "+ str(round(cY)), (int(x)+10,int(y)+0), font, 0.5, (255, 0, 0),(0))
        
        circareatot = circareatot + circarea
        rectareatot = rectareatot + rectarea
        minrectareatot = minrectareatot + minrectarea
        contareatot = contareatot + contarea

    resize = cv2.resize(img,(1200,1200), interpolation = cv2.INTER_AREA)
    cv2.imshow("Shapes", resize)
    cv2.waitKey(500)

    print('convex: ' + str(concavetot))
    print('concave: ' + str(convextot))
    print('minrectareatot: ' + str(minrectareatot))
    print('circareatot: ' + str(circareatot))
    print('iteration number: ' + str(q))
    print(str(circareatot))
    
    #fills the empty matrices
    areacirc.append(circareatot)
    arearec.append(rectareatot)
    areaminrec.append(minrectareatot)
    areacont.append(contareatot)
    concavem.append(concavetot)
    convexm.append(convextot)
    
    #reset variables to zero 
    circareatot = 0
    circareatot = 0
    minrectareatot = 0
    contareatot = 0
    concavetot = 0
    convextot = 0

contvcirc = numpy.divide(areacont,areacirc)
contvrec = numpy.divide(areacont,arearec)
contvminrec = numpy.divide(areacont,areaminrec)
numpy.savetxt('contvminrec',contvminrec, delimiter = ',')
#numpy.savetxt('contvrec',contvrec, delimiter = ',')
numpy.savetxt('contvcirc',contvcirc, delimiter = ',')

#plt.plot(contvcirc, label = 'Contour/Circle')
#plt.plot(contvrec, label = 'Contour/Rectangle')
plt.plot(contvminrec, label = 'Contour/MinRectangle')
plt.legend()
plt.xlabel("Image Slice Number")
plt.ylabel("Area Ratio")
cv2.destroyAllWindows()


x1 = numpy.arange(len(contvcirc))
x2 = numpy.arange(len(convexm))
x3 = numpy.arange(len(concavem))


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x1, contvcirc)
axs[0, 0].set_title('Contvcirc')
axs[0, 1].plot(x1, contvminrec, 'tab:orange')
axs[0, 1].set_title('Contvrec')
axs[1, 0].plot(x2, convexm, 'tab:green')
axs[1, 0].set_title('Convex')
axs[1, 1].plot(x3, concavem, 'tab:red')
axs[1, 1].set_title('Concave')
    
for ax in axs.flat:
    ax.set(xlabel='Image Number', ylabel='y-label')
    
for ax in axs.flat:
    ax.label_outer()
