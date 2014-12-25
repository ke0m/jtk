'''
Created on May 4, 2014

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


def java_smoother(alpha):
    
     #####Getting the Tensors#####
    iimg, s1, s2 = readPnzImage() # this comes from tensors.py
    iimg = mul(0.0001,iimg)
    simgc = zerofloat(len(s1.getValues()), len(s2.getValues()));

    lof = LocalOrientFilter(4.0)
    s = lof.applyForTensors(iimg)
    d00 = EigenTensors2(s); d00.invertStructure(0.0,0.0)
    d01 = EigenTensors2(s); d01.invertStructure(0.0,1.0)
    d02 = EigenTensors2(s); d02.invertStructure(0.0,2.0)
    d04 = EigenTensors2(s); d04.invertStructure(0.0,4.0)
    d11 = EigenTensors2(s); d11.invertStructure(1.0,1.0)
    d12 = EigenTensors2(s); d12.invertStructure(1.0,2.0)
    d14 = EigenTensors2(s); d14.invertStructure(1.0,4.0)
    
    smooth_cpu = LocalSmoothingFilter(0, int(5*sqrt(2*alpha)))
    
    for i in xrange(50):
        smooth_cpu.apply(d04, alpha, iimg, simgc)
        smooth_cpu.applySmoothS(simgc, simgc);
        
    plotPnz("", simgc, s1, s2, dscale=1, png="cpudemo") # This alos comes from tensors.py
        
        
    
    return 0
