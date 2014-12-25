'''
Created on May 5, 2014

@author: Joe Jennings
'''


from java.awt import *
from java.lang import *
from java.nio import *
from javax.swing import *
from java.util import *
	

from edu.mines.jtk.awt import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.util.ArrayMath import *

from tensors import *


def jocl_smoother(alpha):
    
    kernel = LocalDiffusionKernel(LocalDiffusionKernel.Stencil.D22CL) ## This Stencil can vary
    smooth_gpu = LocalSmoothingFilter(0, int(5*sqrt(2*alpha)), kernel)

	
     #####Getting the Tensors#####
    iimg, s1, s2 = readPnzImage() #this comes from tensors.py
    iimg = mul(0.0001,iimg)
    simgg = zerofloat(len(s1.getValues()), len(s2.getValues()));


    lof = LocalOrientFilter(4.0)
    s = lof.applyForTensors(iimg)
    d00 = EigenTensors2(s); d00.invertStructure(0.0,0.0)
    d01 = EigenTensors2(s); d01.invertStructure(0.0,1.0)
    d02 = EigenTensors2(s); d02.invertStructure(0.0,2.0)
    d04 = EigenTensors2(s); d04.invertStructure(0.0,4.0)
    d11 = EigenTensors2(s); d11.invertStructure(1.0,1.0)
    d12 = EigenTensors2(s); d12.invertStructure(1.0,2.0)
    d14 = EigenTensors2(s); d14.invertStructure(1.0,4.0)

    for i in xrange(50):
        smooth_gpu.applyGPU(d04, alpha, iimg, simgg)
        smooth_gpu.applySmoothS(simgg, simgg)
		
        
    plotPnz("", simgg, s1, s2, dscale=1, png="gpudemo") #this also comes from tensors.py
