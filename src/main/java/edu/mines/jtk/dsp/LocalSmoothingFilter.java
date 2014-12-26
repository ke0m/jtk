/****************************************************************************
Copyright (c) 2008, Colorado School of Mines and others. All rights reserved.
This program and accompanying materials are made available under the terms of
the Common Public License - v1.0, which accompanies this distribution, and is 
available at http://www.eclipse.org/legal/cpl-v10.html
****************************************************************************/
package edu.mines.jtk.dsp;

import java.util.logging.Logger;

import edu.mines.jtk.util.Parallel;
import static edu.mines.jtk.util.ArrayMath.*;
import edu.mines.jtk.util.CLUtil;

import org.jocl.*;
import static org.jocl.CL.*;

/**
 * Local smoothing of images with tensor filter coefficients.
 * Smoothing is performed by solving a sparse symmetric positive-definite
 * (SPD) system of equations: (I+G'DG)y = x, where G is a gradient operator, 
 * D is an SPD tensor field, x is an input image, and y is an output image.
 * <p>
 * The sparse system of filter equations (I+G'DG)y = x is solved iteratively, 
 * beginning with y = x. Iterations continue until either the error in the 
 * solution y is below a specified threshold or the number of iterations 
 * exceeds a specified limit.
 * <p>
 * For low wavenumbers the output of this filter approximates the solution 
 * to an anisotropic inhomogeneous diffusion equation, where the filter 
 * input x corresponds to the initial condition at time t = 0 and filter 
 * output y corresponds to the solution at some later time t.
 * <p>
 * Additional smoothing filters may be applied to the input image x before 
 * or after solving the sparse system of equations for the smoothed output 
 * image y. These additional filters compensate for deficiencies in the 
 * gradient operator G, which is a finite-difference approximation that is 
 * poor for high wavenumbers near the Nyquist limit. The extra smoothing
 * filters attenuate these high wavenumbers.
 * <p> 
 * The additional smoothing filter S is a simple 3x3 (or, in 3D, 3x3x3) 
 * weighted-average filter that zeros Nyquist wavenumbers. This filter 
 * is fast and has non-negative coefficients. However, it may smooth too 
 * much, as it attenuates all non-zero wavenumbers, not only the highest
 * wavenumbers. Moreover, this filter is not isotropic. 
 * <p>
 * The other additional smoothing operator L is an isotropic low-pass 
 * filter designed to pass wavenumbers up to a specified maximum.
 * Although slower than S, the cost of applying L to the input image x is 
 * likely to be insignificant relative to the cost of solving the sparse 
 * system of equations for the output image y.
 *
 * @author Dave Hale, Colorado School of Mines
 * @version 2009.12.31
 */
public class LocalSmoothingFilter {
	
  /**
   * Constructs a local smoothing filter with default parameters.
   * The default parameter small is 0.01 and the default maximum 
   * number of iterations is 100. Uses a default 2x2 stencil for the 
   * derivatives in the operator G.
   */
  public LocalSmoothingFilter() {
    this(0.01,100);
  }

  /**
   * Constructs a local smoothing filter with specified iteration parameters.
   * Uses a default 2x2 stencil for the derivatives in the operator G.
   * @param small stop when norm of residuals is less than this factor 
   *  times the norm of the input array.
   * @param niter stop when number of iterations exceeds this limit.
   */
  public LocalSmoothingFilter(double small, int niter) {
    _small = (float)small;
    _niter = niter;
    _ldk = new LocalDiffusionKernel(LocalDiffusionKernel.Stencil.D22);
  }

  /**
   * Constructs a local smoothing filter with specified parameters.
   * @param small stop when norm of residuals is less than this factor 
   *  times the norm of the input array.
   * @param niter stop when number of iterations exceeds this limit.
   * @param ldk the local diffusion kernel that computes y += (I+G'DG)x.
   */
  public LocalSmoothingFilter(
    double small, int niter, LocalDiffusionKernel ldk)
  {
    _small = (float)small;
    _niter = niter;
    _ldk = ldk;
  }

  /**
   * Sets the use of a preconditioner in this local smoothing filter.
   * A preconditioner requires extra memory and more computing time
   * per iteration, but may result in fewer iterations.
   * The default is to not use a preconditioner.
   * @param pc true, to use a preconditioner; false, otherwise.
   */
  public void setPreconditioner(boolean pc) {
    _pc = pc;
  }

  /**
   * Applies this filter for specified constant scale factor.
   * Local smoothing for 1D arrays is a special case that requires no tensors. 
   * All tensors are implicitly scalar values equal to one, so that filtering 
   * is determined entirely by the specified constant scale factor.
   * @param c constant scale factor.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float c, float[] x, float[] y) {
    apply(c,null,x,y);
  }

  /**
   * Applies this filter for specified scale factors.
   * Local smoothing for 1D arrays is a special case that requires no tensors. 
   * All tensors are implicitly scalar values equal to one, so that filtering 
   * is determined entirely by the specified scale factors.
   * @param c constant scale factor.
   * @param s array of scale factors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float c, float[] s, float[] x, float[] y) {
    int n1 = x.length;

    // Sub-diagonal e of SPD tridiagonal matrix I+G'DG; e[0] = e[n1] = 0.0.
    float[] e = new float[n1+1];
    if (s!=null) {
      c = -0.5f*c;
      for (int i1=1; i1<n1; ++i1)
        e[i1] = c*(s[i1]+s[i1-1]);
    } else {
      c = -c;
      for (int i1=1; i1<n1; ++i1)
        e[i1] = c;
    }

    // Work array w overwrites sub-diagonal array e.
    float[] w = e;

    // Solve tridiagonal system of equations (I+G'DG)y = x.
    float t = 1.0f-e[0]-e[1];
    y[0] = x[0]/t;
    for (int i1=1; i1<n1; ++i1) {
      float di = 1.0f-e[i1]-e[i1+1]; // diagonal element
      float ei = e[i1]; // sub-diagonal element
      w[i1] = ei/t;
      t = di-ei*w[i1];
      y[i1] = (x[i1]-ei*y[i1-1])/t;
    }
    for (int i1=n1-1; i1>0; --i1)
      y[i1-1] -= w[i1]*y[i1];
  }

  /**
   * Applies this filter for identity tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float[][] x, float[][] y) 
  {
    apply(null,1.0f,null,x,y);
  }
 
  /**
   * Applies this filter for identity tensors on the GPU
   * @param x input array.
   * @param y output array.
   */	
  public void applyGPU(float[][] x, float[][] y) 
  {
    applyGPU(null,1.0f,null,x,y);
  }
  
  /**
   * Applies this filter for identity tensors and specified scale factor.
   * @param c constant scale factor.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float c, float[][] x, float[][] y) {
    apply(null,c,null,x,y);
  }
  
  
  /**
   * Applies this filter for identity tensors and specified scale factor on the GPU.
   * @param c constant scale factor.
   * @param x input array.
   * @param y output array.
   */
  public void applyGPU(float c, float[][] x, float[][] y) {
	    applyGPU(null,c,null,x,y);
   }

  /**
   * Applies this filter for identity tensors and specified scale factors.
   * @param c constant scale factor.
   * @param s array of scale factors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float c, float[][] s, float[][] x, float[][] y) {
    apply(null,c,s,x,y);
  }

  /**
   * Applies this filter for specified tensors.
   * @param d tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(Tensors2 d, float[][] x, float[][] y) 
  {
    apply(d,1.0f,null,x,y);
  }

  /**
   * Applies this filter for specified tensors and scale factor.
   * @param d tensors.
   * @param c constant scale factor for tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(Tensors2 d, float c, float[][] x, float[][] y) {
    apply(d,c,null,x,y);
  }

  /**
   * Applies this filter for specified tensors and scale factor on the GPU.
   * @param d tensors
   * @param c constant scale factor
   * @param x input array
   * @param y output array
   */
  public void applyGPU(Tensors2 d, float c, float[][] x, float[][] y) {
    applyGPU(d,c,null,x,y);
  }

  /**
   * Applies this filter for specified tensors and scale factors.
   * @param d tensors.
   * @param c constant scale factor for tensors.
   * @param s array of scale factors for tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(
    Tensors2 d, float c, float[][] s, float[][] x, float[][] y) 
  {
    Operator2 a = new A2(_ldk,d,c,s);
    scopy(x,y);
    if (_pc) {
      Operator2 m = new M2(d,c,s,x);
      solve(a,m,x,y);
    } else {
      solve(a,x,y);
    }
  }
  
  /**
   * Applies this filter for specified tensors and scale factors on the GPU.
   * @param d tensors.
   * @param c constant scale factor for tensors.
   * @param s array of scale factors for tensors.
   * @param x input array.
   * @param y output array.
   */
  public void applyGPU(
    Tensors2 d, float c, float[][] s, float[][] x, float[][] y)
  {
    int n1 = x.length;
    int n2 = x[0].length;
    int size = n1*n2;
    int size_y = (n1+1)*(n2+1);
    float[] x1 = new float[size];
    float[] y1 = new float[size_y];
    float[] s1 = new float[size];
    float[] d11 = new float[size];
    float[] d12 = new float[size];
    float[] d22 = new float[size];
    float c1 = 0.25f*c;
    String[] kernelNames = {"clcopy","soSmoothingNew","clsaxpy","cldot","clsxpay"};
    CLUtil.clInit(sourceStr, kernelNames);
    int num_groups = (int) (size/CLUtil.maxWorkGroupSize/4);
    CLUtil.packArray(n1, n2, x, x1); 
    CLUtil.packArray(n1, n2, s, s1);
    scopy(x1,y1);
    CLUtil.unPackTensor(n1, n2, d, d11, d12, d22);
    cl_mem d_x = CLUtil.createGPUBuffer(size, "r");
    cl_mem d_d11 = CLUtil.createGPUBuffer(size, "r");
    cl_mem d_d12 = CLUtil.createGPUBuffer(size, "r");
    cl_mem d_d22 = CLUtil.createGPUBuffer(size, "r");
    cl_mem d_y = CLUtil.createGPUBuffer(size_y, "rw");
    cl_mem d_d = CLUtil.createGPUBuffer(size, "rw");
    cl_mem d_r = CLUtil.createGPUBuffer(size, "rw");
    cl_mem d_q = CLUtil.createGPUBuffer(size, "rw");
    cl_mem d_delta = CLUtil.createGPUBuffer(num_groups, "rw");
    CLUtil.copyToBuffer(d11, d_d11, size);
    CLUtil.copyToBuffer(d12, d_d12, size);
    CLUtil.copyToBuffer(d22, d_d22, size);
    CLUtil.copyToBuffer(x1, d_x, size);
    CLUtil.copyToBuffer(y1, d_y, size_y);
    CLUtil.setKernelArg(CLUtil.kernels[1], d_d11, 1);
    CLUtil.setKernelArg(CLUtil.kernels[1], d_d12, 2);
    CLUtil.setKernelArg(CLUtil.kernels[1], d_d22, 3);
    CLUtil.setKernelArg(CLUtil.kernels[1], c1, 5); 
    Operator2G a = new A2G(_ldk,d11,d12,d22,c,s1); //constructs the diffusion operator
    solveG(a,n1,n2,d_x,d_y,d_d,d_q,d_r,d_delta); //solves the system of equations via a CG solver
    CLUtil.readFromBuffer(d_y, y1, size_y);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseMemObject(d_d);
    clReleaseMemObject(d_r);
    clReleaseMemObject(d_q);
    clReleaseMemObject(d_d11);
    clReleaseMemObject(d_d12);
    clReleaseMemObject(d_d22);
    clReleaseMemObject(d_delta);
    CLUtil.clRelease();
    CLUtil.unPackArray(n1, n2, y1, y);

  }

  /**
   * Applies this filter for identity tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float[][][] x, float[][][] y) 
  {
    apply(null,1.0f,null,x,y);
  }

  /**
   * Applies this filter for identity tensors and specified scale factor.
   * @param c constant scale factor.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float c, float[][][] x, float[][][] y) {
    apply(null,c,null,x,y);
  }

  /**
   * Applies this filter for identity tensors and specified scale factors.
   * @param c constant scale factor.
   * @param s array of scale factors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(float c, float[][][] s, float[][][] x, float[][][] y) {
    apply(null,c,s,x,y);
  }

  /**
   * Applies this filter for specified tensors.
   * @param d tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(Tensors3 d, float[][][] x, float[][][] y) 
  {
    apply(d,1.0f,null,x,y);
  }

  /**
   * Applies this filter for specified tensors and scale factor.
   * @param d tensors.
   * @param c constant scale factor for tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(Tensors3 d, float c, float[][][] x, float[][][] y) {
    apply(d,c,null,x,y);
  }

  /**
   * Applies this filter for specified tensors and scale factors.
   * @param d tensors.
   * @param c constant scale factor for tensors.
   * @param s array of scale factors for tensors.
   * @param x input array.
   * @param y output array.
   */
  public void apply(
    Tensors3 d, float c, float[][][] s, float[][][] x, float[][][] y) 
  {
    Operator3 a = new A3(_ldk,d,c,s);
    scopy(x,y);
    if (_pc) {
      Operator3 m = new M3(d,c,s,x);
      solve(a,m,x,y);
    } else {
      solve(a,x,y);
    }
  }

  /**
   * Applies a simple 3x3 weighted-average smoothing filter S.
   * Input and output arrays x and y may be the same array.
   * @param x input array.
   * @param y output array.
   */
  public void applySmoothS(float[][] x, float[][] y) {
    smoothS(x,y);
  }

  /**
   * Applies a simple 3x3x3 weighted-average smoothing filter S.
   * Input and output arrays x and y may be the same array.
   * @param x input array.
   * @param y output array.
   */
  public void applySmoothS(float[][][] x, float[][][] y) {
    smoothS(x,y);
  }

  /**
   * Applies an isotropic low-pass smoothing filter L.
   * Input and output arrays x and y may be the same array.
   * @param kmax maximum wavenumber not attenuated, in cycles/sample.
   * @param x input array.
   * @param y output array.
   */
  public void applySmoothL(double kmax, float[][] x, float[][] y) {
    smoothL(kmax,x,y);
  }

  /**
   * Applies an isotropic low-pass smoothing filter L.
   * Input and output arrays x and y may be the same array.
   * @param kmax maximum wavenumber not attenuated, in cycles/sample.
   * @param x input array.
   * @param y output array.
   */
  public void applySmoothL(double kmax, float[][][] x, float[][][] y) {
    smoothL(kmax,x,y);
  }

  ///////////////////////////////////////////////////////////////////////////
  // private

  private static final boolean PARALLEL = true; // false for single-threaded

  private static Logger log = 
    Logger.getLogger(LocalSmoothingFilter.class.getName());

  private float _small; // stop iterations when residuals are small
  private int _niter; // number of iterations
  private boolean _pc; // true, for preconditioned CG iterations
  private LocalDiffusionKernel _ldk; // computes y += (I+G'DG)x
  private BandPassFilter _lpf; // lowpass filter, null until applied
  private double _kmax; // maximum wavenumber for lowpass filter
  
  //String containing OpenCL kernels
  String sourceStr =
		  "__kernel void clcopy(int n1, int n2, __global const float* restrict d_x, __global float* restrict d_y)" +"\n" +
		  "{" + "\n" +
		  "    int i  = get_global_id(0);" + "\n" +
		  "    if(i > n1*n2) return;" + "\n" +
		  "    d_y[i] = d_x[i];" + "\n" +
		  "}" + "\n" +
		  "" +"\n" +
	  	  "__kernel void soSmoothingNew(__global const float* restrict d_r," + "\n" +
	  	  "                                                   __global const float* restrict d_d11," + "\n" +
	  	  "                                                   __global const float* restrict d_d12," + "\n" +
	  	  "                                                   __global const float* restrict d_d22," + "\n" +
	  	  "                                                   __global float* restrict d_s," + "\n" +
	  	  "                                                   float alpha," + "\n" +
	  	  "                                                   int n1," + "\n" +
	  	  "                                                   int n2," + "\n" +
	  	  "                                                   int offsetx," + "\n" +
	  	  "                                                   int offsety)" + "\n" +
	  	  "{" + "\n" +
	  	  "     int g1 = get_global_id(0); int g0 = get_global_id(1);" + "\n" +
	  	  "" + "\n" +
	  	  "		int i1 = g1 * 2 + 1 + offsetx;" + "\n" +
	  	  "		int i2 = g0 * 2  + 1 + offsety;" + "\n" +
	  	  "" + "\n" +
	  	  "     if (i1 >= n2) return;" + "\n" +
	  	  "     if (i2 >= n1) return;" + "\n" +
	  	  "" + "\n" + 
	  	  "		float e11, e12, e22, r00, r01, r10, r11, rs, ra, rb, r1, r2, s_1, s_2, s_a, s_b;" + "\n" +
	  	  "" + "\n" + 
	  	  "		e11 = alpha * d_d11[i2*n2 + i1];" + "\n" +
	  	  "		e12 = alpha * d_d12[i2*n2 + i1];" + "\n" +
	  	  "		e22 = alpha * d_d22[i2*n2 + i1];" + "\n" +
	  	  "		r00 = d_r[i2*n2+i1];" + "\n" + 
	  	  "		r01 = d_r[i2*n2+(i1-1)];" + "\n" + 
	  	  "		r10 = d_r[(i2-1)*n2 + i1];" + "\n" +
	  	  "		r11 = d_r[(i2-1)*n2 + (i1-1)];" + "\n" +
	  	  "		rs = 0.25f*(r00+r01+r10+r11);" + "\n" +
	  	  "		ra = r00-r11;" + "\n" +
	  	  "		rb = r01-r10;" + "\n" +
	  	  "		r1 = ra-rb;" + "\n" +
	  	  "		r2 = ra+rb;" + "\n" +
	  	  "		s_1 = e11*r1+e12*r2;" + "\n" +
	  	  "		s_2 = e12*r1+e22*r2;" + "\n" +
	  	  "		s_a = s_1+s_2;" + "\n" +
	  	  "		s_b = s_1-s_2;" + "\n" +
	  	  "		d_s[i2*n2 + i1] += s_a;" + "\n" +
	  	  "		d_s[i2*n2 + (i1 -1)] -= s_b;" + "\n" +
	  	  "		d_s[(i2-1)*n2 + i1] += s_b;" + "\n" +
	  	  "		d_s[(i2-1)*n2 + (i1-1)] -= s_a;" + "\n" +
	  	  "" + "\n" +
	  	  "" + "\n" + 
	  	  "}" + "\n" +
	  	  "" + "\n" +
	  	  "__kernel void clsaxpy(int n1, int n2, float a, __global const float* restrict d_x, __global float* restrict d_y)" + "\n" +
	  	  "{" + "\n" +
	  	  "     int i = get_global_id(0);" + "\n" +
	  	  "     if(i > n1*n2) return;" + "\n" +
	  	  "     d_y[i] += a*d_x[i];" + "\n" +
	  	  "}" + "\n" +
	  	  "" + "\n" +
	  	  "__kernel void cldot(int n1, int n2, __global float4* a_vec, __global float4* b_vec, __global float* output, __local float4* partial_dot) { " +"\n" +
 		  "" + "\n" +
		  "    int gid = get_global_id(0);" + "\n" +
          "    int lid = get_local_id(0);" + "\n" +
          "    int group_size = get_local_size(0);" + "\n" + 
          "    if(gid > n1*n2) return;" + "\n" +
          "" + "\n" +
          "    /* Place product of global values into local memory */" + "\n" +
          "    partial_dot[lid] = a_vec[gid] * b_vec[gid];" + "\n" +
          "    barrier(CLK_LOCAL_MEM_FENCE);" + "\n" +
          "" + "\n" +
          "    /* Repeatedly add values in local memory */" + "\n" +
          "    for(int i = group_size/2; i>0; i >>= 1) {" + "\n" +
          "       if(lid < i) {" + "\n" +
          "           partial_dot[lid] += partial_dot[lid + i];" + "\n" +
          "        }" + "\n" +
          "    barrier(CLK_LOCAL_MEM_FENCE);" + "\n" +
          "    }" + "\n" +
          "" + "\n" +
          "/* Transfer final result to global memory */" + "\n" +
          "   if(lid == 0) {" + "\n" +
          "       output[get_group_id(0)] = dot(partial_dot[0], (float4)(1.0f));" + "\n" +
          "   }" + "\n" +
          "}" + "\n" +
          "__kernel void clsxpay(int n1, int n2, float a, __global const float* restrict x, __global float* restrict y)" + "\n" +
          "{" + "\n" +
          "   int i = get_global_id(0);" + "\n" +
          "   if(i > n1*n2) return;" + "\n" +
          "" + "\n" +
          "   y[i] = a*y[i] + x[i];" + "\n" +
          "}";

  /*
   * A symmetric positive-definite operator.
   */
  private static interface Operator2 {
    public void apply(float[][] x, float[][] y);
  }
  private static interface Operator2G {
	  public void applyGPU(int n1, int n2, cl_mem d_x, cl_mem d_y);
  }
  private static interface Operator3 {
    public void apply(float[][][] x, float[][][] y);
  }

  private static class A2 implements Operator2 {
    A2(LocalDiffusionKernel ldk, Tensors2 d, float c, float[][] s) {
      _ldk = ldk;
      _d = d;
      _c = c;
      _s = s;
    }
    public void apply(float[][] x, float[][] y) {
      scopy(x,y);
      _ldk.apply(_d,_c,_s,x,y);
    }
    private LocalDiffusionKernel _ldk;
    private Tensors2 _d;
    private float _c;
    private float[][] _s;
  }

  private static class M2 implements Operator2 {
    M2(Tensors2 d, float c, float[][] s, float[][] x)  {
      int n1 = x[0].length;
      int n2 = x.length;
      _p = fillfloat(1.0f,n1,n2);
      c *= 0.25f;
      float[] di = new float[3];
      for (int i2=1,m2=0; i2<n2; ++i2,++m2) {
        for (int i1=1,m1=0; i1<n1; ++i1,++m1) {
          float si = s!=null?s[i2][i1]:1.0f;
          float csi = c*si;
          float d11 = csi;
          float d12 = 0.0f;
          float d22 = csi;
          if (d!=null) {
            d.getTensor(i1,i2,di);
            d11 = di[0]*csi;
            d12 = di[1]*csi;
            d22 = di[2]*csi;
          }
          _p[i2][i1] += (d11+d12)+( d12+d22);
          _p[m2][m1] += (d11+d12)+( d12+d22);
          _p[i2][m1] += (d11-d12)+(-d12+d22);
          _p[m2][i1] += (d11-d12)+(-d12+d22);
        }
      }
      div(1.0f,_p,_p);
    }
    public void apply(float[][] x, float[][] y) {
      sxy(_p,x,y);
    }
    private float[][] _p;
  }
  
  private static class A2G implements Operator2G {
    A2G(LocalDiffusionKernel ldk, float[] d11, float[] d12, float[] d22, float c, float[] s) {
      _ldk = ldk;
      _d11 = d11;
      _d12 = d12;
      _d22 = d22;
      _c = c;
      _s = s;

    }

    public void applyGPU(int n1, int n2, cl_mem d_x, cl_mem d_y)
    {
      long[] global_work_group_size = {n1*n2};
      CLUtil.setKernelArg(CLUtil.kernels[0], n1, 0);
      CLUtil.setKernelArg(CLUtil.kernels[0], n2, 1);
      CLUtil.setKernelArg(CLUtil.kernels[0], d_x, 2);
      CLUtil.setKernelArg(CLUtil.kernels[0], d_y, 3);
      CLUtil.executeKernel(CLUtil.kernels[0], 1, global_work_group_size);
      _ldk.applyGPU(n1, n2);

    }

    private LocalDiffusionKernel _ldk;
    private float[] _d11;
    private float[] _d12;
    private float[] _d22;
    private float _c;
    private float[] _s;
  }

  private static class A3 implements Operator3 {
    A3(LocalDiffusionKernel ldk, Tensors3 d, float c, float[][][] s) {
      _ldk = ldk;
      _d = d;
      _c = c;
      _s = s;
    }
    public void apply(float[][][] x, float[][][] y) {
      scopy(x,y);
      _ldk.apply(_d,_c,_s,x,y);
    }
    private LocalDiffusionKernel _ldk;
    private Tensors3 _d;
    private float _c;
    private float[][][] _s;
  }

  private static class M3 implements Operator3 {
    M3(Tensors3 d, float c, float[][][] s, float[][][] x)  {
      int n1 = x[0][0].length;
      int n2 = x[0].length;
      int n3 = x.length;
      _p = fillfloat(1.0f,n1,n2,n3);
      c *= 0.0625f;
      float[] di = new float[6];
      for (int i3=1,m3=0; i3<n3; ++i3,++m3) {
        for (int i2=1,m2=0; i2<n2; ++i2,++m2) {
          for (int i1=1,m1=0; i1<n1; ++i1,++m1) {
            float si = s!=null?s[i3][i2][i1]:1.0f;
            float csi = c*si;
            float d11 = csi;
            float d12 = 0.0f;
            float d13 = 0.0f;
            float d22 = csi;
            float d23 = 0.0f;
            float d33 = csi;
            if (d!=null) {
              d.getTensor(i1,i2,i3,di);
              d11 = di[0]*csi;
              d12 = di[1]*csi;
              d13 = di[2]*csi;
              d22 = di[3]*csi;
              d23 = di[4]*csi;
              d33 = di[5]*csi;
            }
            _p[i3][i2][i1] += ( d11+d12+d13)+( d12+d22+d23)+( d13+d23+d33);
            _p[m3][m2][m1] += ( d11+d12+d13)+( d12+d22+d23)+( d13+d23+d33);
            _p[i3][m2][i1] += ( d11-d12+d13)+(-d12+d22-d23)+( d13-d23+d33);
            _p[m3][i2][m1] += ( d11-d12+d13)+(-d12+d22-d23)+( d13-d23+d33);
            _p[m3][i2][i1] += ( d11+d12-d13)+( d12+d22-d23)+(-d13-d23+d33);
            _p[i3][m2][m1] += ( d11+d12-d13)+( d12+d22-d23)+(-d13-d23+d33);
            _p[m3][m2][i1] += ( d11-d12-d13)+(-d12+d22+d23)+(-d13+d23+d33);
            _p[i3][i2][m1] += ( d11-d12-d13)+(-d12+d22+d23)+(-d13+d23+d33);
          }
        }
      }
      div(1.0f,_p,_p);
    }
    public void apply(float[][][] x, float[][][] y) {
      sxy(_p,x,y);
    }
    private float[][][] _p;
  }

  /*
   * Computes y = lowpass(x). Arrays x and y may be the same array.
   */
  private void smoothL(double kmax, float[][] x, float[][] y) {
    ensureLowpassFilter(kmax);
    _lpf.apply(x,y);
  }
  private void smoothL(double kmax, float[][][] x, float[][][] y) {
    ensureLowpassFilter(kmax);
    _lpf.apply(x,y);
  }
  private void ensureLowpassFilter(double kmax) {
    if (_lpf==null || _kmax!=kmax) {
      _kmax = kmax;
      double kdelta = 0.5-kmax;
      double kupper = kmax+0.5*kdelta;
      _lpf = new BandPassFilter(0.0,kupper,kdelta,0.01);
      _lpf.setExtrapolation(BandPassFilter.Extrapolation.ZERO_SLOPE);
      _lpf.setFilterCaching(false);
    }
  }

  /*
   * Computes y = S'Sx. Arrays x and y may be the same array.
   */
  private static void smoothS(float[][] x, float[][] y) {
    int n1 = x[0].length;
    int n2 = x.length;
    int n1m = n1-1;
    int n2m = n2-1;
    float[][] t = new float[3][n1];
    scopy(x[0],t[0]);
    scopy(x[0],t[1]);
    for (int i2=0; i2<n2; ++i2) {
      int i2m = (i2>0)?i2-1:0;
      int i2p = (i2<n2m)?i2+1:n2m;
      int j2m = i2m%3;
      int j2  = i2%3;
      int j2p = i2p%3;
      scopy(x[i2p],t[j2p]);
      float[] x2m = t[j2m];
      float[] x2p = t[j2p];
      float[] x20 = t[j2];
      float[] y2 = y[i2];
      for (int i1=0; i1<n1; ++i1) {
        int i1m = (i1>0)?i1-1:0;
        int i1p = (i1<n1m)?i1+1:n1m;
        y2[i1] = 0.2500f*(x20[i1 ]) +
                 0.1250f*(x20[i1m]+x20[i1p]+x2m[i1 ]+x2p[i1 ]) +
                 0.0625f*(x2m[i1m]+x2m[i1p]+x2p[i1m]+x2p[i1p]);
      }
    }
  }

  /*
   * Computes y = S'Sx. Arrays x and y may be the same array.
   */
  private static void smoothS(float[][][] x, float[][][] y) {
    int n1 = x[0][0].length;
    int n2 = x[0].length;
    int n3 = x.length;
    int n1m = n1-1;
    int n2m = n2-1;
    int n3m = n3-1;
    float[][][] t = new float[3][n2][n1];
    scopy(x[0],t[0]);
    scopy(x[0],t[1]);
    for (int i3=0; i3<n3; ++i3) {
      int i3m = (i3>0)?i3-1:0;
      int i3p = (i3<n3m)?i3+1:n3m;
      int j3m = i3m%3;
      int j3  = i3%3;
      int j3p = i3p%3;
      scopy(x[i3p],t[j3p]);
      float[][] x3m = t[j3m];
      float[][] x3p = t[j3p];
      float[][] x30 = t[j3];
      float[][] y30 = y[i3];
      for (int i2=0; i2<n2; ++i2) {
        int i2m = (i2>0)?i2-1:0;
        int i2p = (i2<n2m)?i2+1:n2m;
        float[] x3m2m = x3m[i2m];
        float[] x3m20 = x3m[i2 ];
        float[] x3m2p = x3m[i2p];
        float[] x302m = x30[i2m];
        float[] x3020 = x30[i2 ];
        float[] x302p = x30[i2p];
        float[] x3p2m = x3p[i2m];
        float[] x3p20 = x3p[i2 ];
        float[] x3p2p = x3p[i2p];
        float[] y3020 = y30[i2 ];
        for (int i1=0; i1<n1; ++i1) {
          int i1m = (i1>0)?i1-1:0;
          int i1p = (i1<n1m)?i1+1:n1m;
          y3020[i1] = 0.125000f*(x3020[i1 ]) +
                      0.062500f*(x3020[i1m]+x3020[i1p]+
                                 x302m[i1 ]+x302p[i1 ]+
                                 x3m20[i1 ]+x3p20[i1 ]) +
                      0.031250f*(x3m20[i1m]+x3m20[i1p]+
                                 x3m2m[i1 ]+x3m2p[i1 ]+
                                 x302m[i1m]+x302m[i1p]+
                                 x302p[i1m]+x302p[i1p]+
                                 x3p20[i1m]+x3p20[i1p]+
                                 x3p2m[i1 ]+x3p2p[i1 ]) +
                      0.015625f*(x3m2m[i1m]+x3m2m[i1p]+
                                 x3m2p[i1m]+x3m2p[i1p]+
                                 x3p2m[i1m]+x3p2m[i1p]+
                                 x3p2p[i1m]+x3p2p[i1p]);
        }
      }
    }
  }

  // Conjugate-gradient solution of Ax = b, with no preconditioner.
  // Uses the initial values of x; does not assume they are zero.
  private void solve(Operator2 a, float[][] b, float[][] x) {
    int n1 = b[0].length;
    int n2 = b.length;
    float[][] d = new float[n2][n1];
    float[][] q = new float[n2][n1];
    float[][] r = new float[n2][n1];
    scopy(b,r);
    a.apply(x,q); 
    saxpy(-1.0f,q,r); // r = b-Ax
    scopy(r,d); // d = r
    float delta = sdot(r,r); // delta = r'r
    float bnorm = sqrt(sdot(b,b));
    float rnorm = sqrt(delta);
    float rnormBegin = rnorm;
    float rnormSmall = bnorm*_small;
    int iter;
    log.fine("solve: bnorm="+bnorm+" rnorm="+rnorm);
    //for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
    for (iter=0; iter<_niter; ++iter) {
      log.finer("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      a.apply(d,q); // q = Ad
      float dq = sdot(d,q); // d'q = d'Ad
      float alpha = delta/dq; // alpha = r'r/d'Ad
      saxpy( alpha,d,x); // x = x+alpha*d
      saxpy(-alpha,q,r); // r = r-alpha*q
      float deltaOld = delta;
      delta = sdot(r,r); // delta = r'r
      float beta = delta/deltaOld;
      sxpay(beta,r,d); // d = r+beta*d
      rnorm = sqrt(delta);
    }
    log.fine("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
  }
 
  // Conjugate-gradient solution of Ax = b for the GPU, with no preconditioner.
  // Uses the initial values of x; does not assume they are zero.
  // Note that the convergence of this solver depends only on a set number of iterations 
  // that is specified in the constructor of this class.
  private void solveG(Operator2G a, int n1, int n2, cl_mem d_x, cl_mem d_y, cl_mem d_d, cl_mem d_q, cl_mem d_r, cl_mem d_delta) {
    int size = n1*n2;
    float[] r = new float[size];
    float[] q = new float[size];
    float[] d = new float[size];
    float[] p_delta = new float[(int)(size/CLUtil.maxWorkGroupSize/4)]; 
    long[] local_work_size_vec = {CLUtil.maxWorkGroupSize};
    long[] global_work_size_oned = {size};
    long[] global_work_size_vec = {((long)Math.ceil(size/local_work_size_vec[0]/4)+1)*local_work_size_vec[0]}; 
    CLUtil.setKernelArg(CLUtil.kernels[0], n1, 0);
    CLUtil.setKernelArg(CLUtil.kernels[0], n2, 1);
    CLUtil.setKernelArg(CLUtil.kernels[0], d_x, 2);
    CLUtil.setKernelArg(CLUtil.kernels[0], d_r, 3);
    CLUtil.executeKernel(CLUtil.kernels[0], 1, global_work_size_oned); //clcopy
    CLUtil.setKernelArg(CLUtil.kernels[1], d_y, 0);
    CLUtil.setKernelArg(CLUtil.kernels[1], d_q, 4);
    CLUtil.setKernelArg(CLUtil.kernels[1], n1, 6);
    CLUtil.setKernelArg(CLUtil.kernels[1], n2, 7);
    a.applyGPU(n1,n2,d_x,d_q); //soSmoothingNew
    CLUtil.setKernelArg(CLUtil.kernels[2], n1, 0);
    CLUtil.setKernelArg(CLUtil.kernels[2], n2, 1);
    CLUtil.setKernelArg(CLUtil.kernels[2], -1.0f, 2);
    CLUtil.setKernelArg(CLUtil.kernels[2], d_q, 3);
    CLUtil.setKernelArg(CLUtil.kernels[2], d_r, 4);
    CLUtil.executeKernel(CLUtil.kernels[2], 1, global_work_size_oned); //clsaxpy
    CLUtil.setKernelArg(CLUtil.kernels[0], n1, 0);
    CLUtil.setKernelArg(CLUtil.kernels[0], n2, 1);
    CLUtil.setKernelArg(CLUtil.kernels[0], d_r, 2);
    CLUtil.setKernelArg(CLUtil.kernels[0], d_d, 3);
    CLUtil.executeKernel(CLUtil.kernels[0], 1, global_work_size_oned); //clcopy
    CLUtil.setKernelArg(CLUtil.kernels[3], n1, 0);
    CLUtil.setKernelArg(CLUtil.kernels[3], n2, 1);
    CLUtil.setKernelArg(CLUtil.kernels[3], d_r, 2);
    CLUtil.setKernelArg(CLUtil.kernels[3], d_r, 3);
    CLUtil.setKernelArg(CLUtil.kernels[3], d_delta, 4);
    CLUtil.setLocalKernelArg(CLUtil.kernels[3], (int)(CLUtil.maxWorkGroupSize*4), 5);
    CLUtil.executeKernel(CLUtil.kernels[3], 1, global_work_size_vec, local_work_size_vec); //cldot
    CLUtil.readFromBuffer(d_delta, p_delta, (int)(size/CLUtil.maxWorkGroupSize/4));
    float delta = sum(p_delta);
    CLUtil.setKernelArg(CLUtil.kernels[3], n1, 0);
    CLUtil.setKernelArg(CLUtil.kernels[3], n2, 1);
    CLUtil.setKernelArg(CLUtil.kernels[3], d_x, 2);
    CLUtil.setKernelArg(CLUtil.kernels[3], d_x, 3);
    CLUtil.setKernelArg(CLUtil.kernels[3], d_delta, 4);
    CLUtil.setLocalKernelArg(CLUtil.kernels[3], (int)(CLUtil.maxWorkGroupSize*4), 5);
    CLUtil.executeKernel(CLUtil.kernels[3], 1, global_work_size_vec, local_work_size_vec); //cldot
    CLUtil.readFromBuffer(d_delta, p_delta, (int)(size/CLUtil.maxWorkGroupSize/4));
    float bnorm = sqrt(sum(p_delta));
    int iter;
    for (iter=0; iter<_niter; ++iter) {
      CLUtil.setKernelArg(CLUtil.kernels[1], d_d, 0);
      CLUtil.setKernelArg(CLUtil.kernels[1], d_q, 4);
      CLUtil.setKernelArg(CLUtil.kernels[1], n1, 6);
      CLUtil.setKernelArg(CLUtil.kernels[1], n2, 7);
      a.applyGPU(n1,n2,d_d,d_q); // q = Ad soSmoothingNew
      CLUtil.setKernelArg(CLUtil.kernels[3], n1, 0);
      CLUtil.setKernelArg(CLUtil.kernels[3], n2, 1);
      CLUtil.setKernelArg(CLUtil.kernels[3], d_d, 2);
      CLUtil.setKernelArg(CLUtil.kernels[3], d_q, 3);
      CLUtil.setKernelArg(CLUtil.kernels[3], d_delta, 4);
      CLUtil.setLocalKernelArg(CLUtil.kernels[3], (int)(CLUtil.maxWorkGroupSize*4), 5);
      CLUtil.executeKernel(CLUtil.kernels[3], 1, global_work_size_vec, local_work_size_vec); //cldot
      CLUtil.readFromBuffer(d_delta, p_delta, (int)(size/CLUtil.maxWorkGroupSize/4));
      float dq = sum(p_delta);
      float alpha = delta/dq; // alpha = r'r/d'Ad
      CLUtil.setKernelArg(CLUtil.kernels[2], n1, 0);
      CLUtil.setKernelArg(CLUtil.kernels[2], n2, 1);
      CLUtil.setKernelArg(CLUtil.kernels[2], alpha, 2);
      CLUtil.setKernelArg(CLUtil.kernels[2], d_d, 3);
      CLUtil.setKernelArg(CLUtil.kernels[2], d_y, 4);
      CLUtil.executeKernel(CLUtil.kernels[2], 1, global_work_size_oned); //clsaxpy
      CLUtil.setKernelArg(CLUtil.kernels[2], n1, 0);
      CLUtil.setKernelArg(CLUtil.kernels[2], n2, 1);
      CLUtil.setKernelArg(CLUtil.kernels[2], -alpha, 2);
      CLUtil.setKernelArg(CLUtil.kernels[2], d_q, 3);
      CLUtil.setKernelArg(CLUtil.kernels[2], d_r, 4);
      CLUtil.executeKernel(CLUtil.kernels[2], 1, global_work_size_oned); //clsaxpy
      float deltaOld = delta;
      CLUtil.setKernelArg(CLUtil.kernels[3], n1, 0);
      CLUtil.setKernelArg(CLUtil.kernels[3], n2, 1);
      CLUtil.setKernelArg(CLUtil.kernels[3], d_r, 2);
      CLUtil.setKernelArg(CLUtil.kernels[3], d_r, 3);
      CLUtil.setKernelArg(CLUtil.kernels[3], d_delta, 4);
      CLUtil.setLocalKernelArg(CLUtil.kernels[3], (int)(CLUtil.maxWorkGroupSize*4), 5);
      CLUtil.executeKernel(CLUtil.kernels[3], 1, global_work_size_vec, local_work_size_vec); //cldot
      CLUtil.readFromBuffer(d_delta, p_delta, (int)(size/CLUtil.maxWorkGroupSize/4));
      delta = sum(p_delta);
      float beta = delta/deltaOld; 
      CLUtil.setKernelArg(CLUtil.kernels[4], n1, 0);
      CLUtil.setKernelArg(CLUtil.kernels[4], n2, 1);
      CLUtil.setKernelArg(CLUtil.kernels[4], beta, 2);
      CLUtil.setKernelArg(CLUtil.kernels[4], d_r, 3);
      CLUtil.setKernelArg(CLUtil.kernels[4], d_d, 4);
      CLUtil.executeKernel(CLUtil.kernels[4], 1, global_work_size_oned); //clsxpay
    }
  }
  
  private void solve(Operator3 a, float[][][] b, float[][][] x) {
    int n1 = b[0][0].length;
    int n2 = b[0].length;
    int n3 = b.length;
    float[][][] d = new float[n3][n2][n1];
    float[][][] q = new float[n3][n2][n1];
    float[][][] r = new float[n3][n2][n1];
    scopy(b,r); a.apply(x,q); saxpy(-1.0f,q,r); // r = b-Ax
    scopy(r,d);
    float delta = sdot(r,r);
    float bnorm = sqrt(sdot(b,b));
    float rnorm = sqrt(delta);
    float rnormBegin = rnorm;
    float rnormSmall = bnorm*_small;
    int iter;
    log.fine("solve: bnorm="+bnorm+" rnorm="+rnorm);
    for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
      log.finer("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      a.apply(d,q);
      float dq = sdot(d,q);
      float alpha = delta/dq;
      saxpy( alpha,d,x);
      if (iter%100<99) {
        saxpy(-alpha,q,r);
      } else {
        scopy(b,r); a.apply(x,q); saxpy(-1.0f,q,r);
      }
      float deltaOld = delta;
      delta = sdot(r,r);
      float beta = delta/deltaOld;
      sxpay(beta,r,d);
      rnorm = sqrt(delta);
    }
    log.fine("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
  }

  // Conjugate-gradient solution of Ax = b, with preconditioner M.
  // Uses the initial values of x; does not assume they are zero.
  private void solve(Operator2 a, Operator2 m, float[][] b, float[][] x) {
    int n1 = b[0].length;
    int n2 = b.length;
    float[][] d = new float[n2][n1];
    float[][] q = new float[n2][n1];
    float[][] r = new float[n2][n1];
    float[][] s = new float[n2][n1];
    scopy(b,r);
    a.apply(x,q);
    saxpy(-1.0f,q,r); // r = b-Ax
    float bnorm = sqrt(sdot(b,b));
    float rnorm = sqrt(sdot(r,r));
    float rnormBegin = rnorm;
    float rnormSmall = bnorm*_small;
    m.apply(r,s); // s = Mr
    scopy(s,d); // d = s
    float delta = sdot(r,s); // r's = r'Mr
    int iter;
    log.fine("msolve: bnorm="+bnorm+" rnorm="+rnorm);
    for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
      log.finer("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      a.apply(d,q); // q = Ad
      float alpha = delta/sdot(d,q); // alpha = r'Mr/d'Ad
      saxpy( alpha,d,x); // x = x+alpha*d
      saxpy(-alpha,q,r); // r = r-alpha*q
      m.apply(r,s); // s = Mr
      float deltaOld = delta;
      delta = sdot(r,s); // delta = r's = r'Mr
      float beta = delta/deltaOld;
      sxpay(beta,s,d); // d = s+beta*d
      rnorm  = sqrt(sdot(r,r));
    }
    log.fine("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
  }
  private void solve(Operator3 a, Operator3 m, float[][][] b, float[][][] x) {
    int n1 = b[0][0].length;
    int n2 = b[0].length;
    int n3 = b.length;
    float[][][] d = new float[n3][n2][n1];
    float[][][] q = new float[n3][n2][n1];
    float[][][] r = new float[n3][n2][n1];
    float[][][] s = new float[n3][n2][n1];
    scopy(b,r); a.apply(x,q); saxpy(-1.0f,q,r); // r = b-Ax
    float bnorm = sqrt(sdot(b,b));
    float rnorm = sqrt(sdot(r,r));
    float rnormBegin = rnorm;
    float rnormSmall = bnorm*_small;
    m.apply(r,s); // s = Mr
    scopy(s,d); // d = s
    float delta = sdot(r,s); // r's = r'Mr
    int iter;
    log.fine("msolve: bnorm="+bnorm+" rnorm="+rnorm);
    for (iter=0; iter<_niter && rnorm>rnormSmall; ++iter) {
      log.finer("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
      a.apply(d,q); // q = Ad
      float alpha = delta/sdot(d,q); // alpha = r'Mr/d'Ad
      saxpy( alpha,d,x); // x = x+alpha*d
      if (iter%100<99) {
        saxpy(-alpha,q,r); // r = r-alpha*q
      } else {
        scopy(b,r); a.apply(x,q); saxpy(-1.0f,q,r); // r = b-Ax
      }
      m.apply(r,s); // s = Mr
      float deltaOld = delta;
      delta = sdot(r,s); // delta = r's = r'Mr
      float beta = delta/deltaOld;
      sxpay(beta,s,d); // d = s+beta*d
      rnorm  = sqrt(sdot(r,r));
    }
    log.fine("  iter="+iter+" rnorm="+rnorm+" ratio="+rnorm/rnormBegin);
  }

  // Zeros array x.
  private static void szero(float[] x) {
    zero(x);
  }
  private static void szero(float[][] x) {
    zero(x);
  }
  private static void szero(final float[][][] x) {
    final int n3 = x.length;
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        szero(x[i3]);
      }
    });
  }

  // Copys array x to array y.
  private static void scopy(float[] x, float[] y) {
    copy(x,y);
  }
  private static void scopy(float[][] x, float[][] y) {
    copy(x,y);
  }
  private static void scopy(final float[][][] x, final float[][][] y) {
    final int n3 = x.length;
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        scopy(x[i3],y[i3]);
      }
    });
  }

  // Returns the dot product x'y.
  private static float sdot(float[][] x, float[][] y) {
    int n1 = x[0].length;
    int n2 = x.length;
    float d = 0.0f;
    for (int i2=0; i2<n2; ++i2) {
      float[] x2 = x[i2], y2 = y[i2];
      for (int i1=0; i1<n1; ++i1) {
        d += x2[i1]*y2[i1];
      }
    }
    return d;
  }
  private static float sdot(final float[][][] x, final float[][][] y) {
    final int n3 = x.length;
    final float[] d3 = new float[n3];
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        d3[i3] = sdot(x[i3],y[i3]);
      }
    });
    float d = 0.0f;
    for (int i3=0; i3<n3; ++i3)
      d += d3[i3];
    return d;
  }

  // Computes y = y + a*x.
  private static void saxpy(float a, float[][] x, float[][] y) {
    int n1 = x[0].length;
    int n2 = x.length;
    for (int i2=0; i2<n2; ++i2) {
      float[] x2 = x[i2], y2 = y[i2];
      for (int i1=0; i1<n1; ++i1) {
        y2[i1] += a*x2[i1];
      }
    }
  }
  private static void saxpy(
    final float a, final float[][][] x, final float[][][] y)
  {
    final int n3 = x.length;
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        saxpy(a,x[i3],y[i3]);
      }
    });
  }

  // Computes y = x + a*y.
  private static void sxpay(float a, float[][] x, float[][] y) {
    int n1 = x[0].length;
    int n2 = x.length;
    for (int i2=0; i2<n2; ++i2) {
      float[] x2 = x[i2], y2 = y[i2];
      for (int i1=0; i1<n1; ++i1) {
        y2[i1] = a*y2[i1]+x2[i1];
      }
    }
  }
  private static void sxpay(
    final float a, final float[][][] x, final float[][][] y)
  {
    final int n3 = x.length;
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        sxpay(a,x[i3],y[i3]);
      }
    });
  }

  // Computes z = x*y.
  private static void sxy(float[][] x, float[][] y, float[][] z) {
    int n1 = x[0].length;
    int n2 = x.length;
    for (int i2=0; i2<n2; ++i2) {
      float[] x2 = x[i2], y2 = y[i2], z2 = z[i2];
      for (int i1=0; i1<n1; ++i1) {
        z2[i1] = x2[i1]*y2[i1];
      }
    }
  }
  private static void sxy(
    final float[][][] x, final float[][][] y, final float[][][] z) 
  {
    final int n3 = x.length;
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
        sxy(x[i3],y[i3],z[i3]);
      }
    });
  }
}
