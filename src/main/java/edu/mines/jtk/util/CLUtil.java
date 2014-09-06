package edu.mines.jtk.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import org.jocl.*;

import static org.jocl.CL.*;
import edu.mines.jtk.dsp.*;

//TODO: 1. Add the capablitity to create multiple kernels from one string (Has this been done? I think so)
//      2. Make this more generic. Multithreading for both CPU and GPU
//		3. Make it generic for any graphics card of any local_group_size (number of threads per block)
//		4. Add in comments for better code readibility


public class CLUtil {

	int numPlatforms;
	int numDevices;
	public static cl_platform_id[] platforms = new cl_platform_id[1];
	public static cl_device_id[] devices = new cl_device_id[1];
	public static cl_context context;
	public static cl_command_queue queue;
	public static cl_mem buff1;
	public static cl_mem buff2;
	public static int err;
	public static int[] errM = new int[1];

	public static cl_program program;
	public static cl_kernel[] kernels;

	String kernelCode;
	String kernelName;
	
//	public  CLUtil(String sourceStr, String nameOfKernel){
//		
//	}
	
	public static void clInit(String sourceStr, String[] kernelNames)
	{
		err = clGetPlatformIDs(1, platforms, null);
		if(err != CL.CL_SUCCESS){
			System.out.println("Error: Failed to locate platform.");
			System.exit(1);
		}
		
		err = clGetDeviceIDs(platforms[0], CL.CL_DEVICE_TYPE_GPU, 1, devices, null);
		if(err != CL_SUCCESS){
			System.out.println("Error: Failed to locate device.");
			System.out.println("OpenCL Error Code: " + err);
			System.exit(1);
		}	
		
		context = clCreateContext(null, 1, devices, null, null, errM);
		if(errM[0] != CL_SUCCESS){
			System.out.println("Error: Failed to create the context.");
			System.out.println("OpenCL Error Code: " + err);
			System.exit(1);
		}
		
		program = clCreateProgramWithSource(context, 1, new String[]{ sourceStr },  null, errM);
		if(errM[0] != CL_SUCCESS){
			System.out.println("Error: Failed to create the program");
			System.exit(1);
		}
	
		err = clBuildProgram(program, 1, devices, "-cl-strict-aliasing -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math -DMAC", null, null);
		if(err != CL_SUCCESS){
			System.out.println("Error: Failed to build the program.");
			System.out.println(getProgramInfo(devices, program));
			System.exit(1);
		}
		
		kernels = new cl_kernel[kernelNames.length];
				
		for(int i = 0; i < kernelNames.length; i++)
		{
			kernels[i] = CL.clCreateKernel(program, kernelNames[i], errM);
			if(errM[0] != CL_SUCCESS){
				System.out.println("Error: Failed to create the kernel.");
				System.out.println("OpenCL Error Code: " + errM[0]);
				System.exit(1);
			}
			
		}

	
		queue = CL.clCreateCommandQueue(context, devices[0], 0, errM);
		if(errM[0] != CL.CL_SUCCESS){
			System.out.println("Error: Failed to create the command queue.");
			System.out.println("OpenCL error code: " + errM[0]);
			System.exit(1);
		}
		
	}
	
	  /**
	   * Reads a text file into a String.
	   * @param file the text file to be read into the String
	   * @return the string containing the text from the text file
	   */
	  public static String readFile(String file) throws FileNotFoundException{
	
		  String text = new Scanner( new File(file)).useDelimiter("\\A").next();
	
		  return text;
	
	  }

	  public static cl_mem createGPUBuffer(int sizeOfData, String rw)
	  {
		  cl_mem buff = null;
		  if(rw == "r")
		  {
			  buff = clCreateBuffer(CLUtil.context, CL_MEM_READ_ONLY, Sizeof.cl_float*sizeOfData, null, CLUtil.errM);
			  if(CLUtil.errM[0] < 0){
				  System.out.println("Error: Could not create the buffer.");
				  System.out.println("OpenCL error code:  " + CLUtil.errM[0]);
				  System.exit(1);
			  }
		  }
		  else if(rw == "w")
		  {
			  buff = clCreateBuffer(CLUtil.context, CL_MEM_WRITE_ONLY, Sizeof.cl_float*sizeOfData, null, CLUtil.errM);
			  if(CLUtil.errM[0] < 0){
				  System.out.println("Error: Could not create the buffer.");
				  System.out.println("OpenCL error code:  " + CLUtil.errM[0]);
				  System.exit(1);
			  }
		  }
		  else
		  {
			  buff = clCreateBuffer(CLUtil.context, CL_MEM_READ_WRITE, Sizeof.cl_float*sizeOfData, null, CLUtil.errM);
			  if(CLUtil.errM[0] < 0){
				  System.out.println("Error: Could not create the buffer.");
				  System.out.println("OpenCL error code:  " + CLUtil.errM[0]);
				  System.exit(1);
			  }
		  }
			return buff;
	  }
	  
	  
	  
	  public static void copyToBuffer(float[] x, cl_mem buff, int sizeOfData)
	  {
			err = clEnqueueWriteBuffer(queue, buff, CL_FALSE, 0, Sizeof.cl_float*sizeOfData, Pointer.to(x), 0, null, null);
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not write data to buffer");
				System.out.println("OpenCL Error Code: " + err);
				System.exit(1);
			}	
	  }
	  
	  public static void readFromBuffer(cl_mem buff, float[] x, int sizeOfData)
	  {
		  err = clEnqueueReadBuffer(queue, buff, CL_TRUE, 0, Sizeof.cl_float*sizeOfData, Pointer.to(x), 0, null, null);
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not read the data from buffer.");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);
			}
	  }
	  
	  public static void setKernelArg(cl_kernel kernel, cl_mem buff, int argNum)
	  {
		  err = clSetKernelArg(kernel, argNum, Sizeof.cl_mem, Pointer.to(buff));
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not set kernel argument.");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);	
			}
	  }
	  
	  public static void setKernelArg(cl_kernel kernel, float x, int argNum)
	  {
		  float[] xp = new float[1];
		  xp[0] = x;
		  err = clSetKernelArg(kernel, argNum, Sizeof.cl_float, Pointer.to(xp));
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not set kernel argument.");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);	
			}
	  }
	
	  public static void setKernelArg(cl_kernel kernel, int x, int argNum)
	  {
		  int[] xp = new int[1];
		  xp[0] = x;
		  err = clSetKernelArg(kernel, argNum, Sizeof.cl_int, Pointer.to(xp));
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not set kernel argument.");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);	
			}
		  
	  }
	  
	  
	  public static void setLocalKernelArg(cl_kernel kernel, int sizeOfData, int argNum)
	  {
		  err = clSetKernelArg(kernel, argNum, sizeOfData * Sizeof.cl_float, null);
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not set kernel argument.");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);	
			}
	  }
	  
	  public static void executeKernel(cl_kernel kernel, int dataDims, long[] global_group_size, long[] local_group_size)
	  {
		  err = clEnqueueNDRangeKernel(queue, kernel, dataDims, null, global_group_size, local_group_size, 0, null, null);
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not execute the kernel");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);
			}
	  }
	  
	  public static void executeKernel(cl_kernel kernel, int dataDims, long[] global_group_size)
	  {
		  err = clEnqueueNDRangeKernel(queue, kernel, dataDims, null, global_group_size, null, 0, null, null);
			if(err != CL.CL_SUCCESS){
				System.out.println("Error: Could not execute the kernel");
				System.out.println("OpenCL error code: " + err);
				System.exit(1);
			}
	  }
	  
	  public static void clRelease()
	  {
		  for(int i = 0; i < kernels.length; i++)
		  {
			  clReleaseKernel(kernels[i]);
		  }
		  
		  clReleaseCommandQueue(queue);
		  clReleaseProgram(program);
		  clReleaseContext(context);
	  }
	  
	  public static void packArray(int n1, int n2, float [][] x, float[] x1)
	  {
		  if(x==null)
		  {
			 return;
		  }
		
		  else
		  {
			  for(int i = 0; i < n1; i++)
			  {
				  for(int j = 0; j < n2; j++)
				  {
					 x1[i*n2 + j] = x[i][j];
				  }
			  }
		  }
	  }
	
	public static void unPackArray(int n1, int n2, float[] x1, float[][] x)
	{
		if(x1 == null)
		{
			return;
		}
		
		for(int i = 0; i < n1; i++)
		{
			for(int j = 0; j < n2; j++)
			{
				x[i][j] = x1[i*n2 + j];
			}
		}
		
	}

	public static void unPackTensor(int n1, int n2, Tensors2 d, float[] d11, float[] d12, float[] d22)
	{
		float[] di = new float[3];
		if(d==null)
		{
			for(int i = 1; i < n1; i++)
			{
				for(int j = 1; j < n2; j++)
				{
					  d11[i*n2 + j] = 1;
					  d12[i*n2 + j] = 0;
					  d22[i*n2 + j] = 1;

				}
			}
		}
		
		else
		{
			for(int i = 1; i < n1; i++)
			{
				for(int j = 1; j < n2; j++)
				{
					  d.getTensor(j, i, di);
					  d11[i*n2 + j] = di[0];
					  d12[i*n2 + j] = di[1];
					  d22[i*n2 + j] = di[2];

				}
			}
		}
		
	}
	/////////////Private Methods
	
	  private static String getProgramInfo(cl_device_id[] devices, cl_program program){
			
			StringBuffer sb = new StringBuffer();
			long[] logSize = new long[1];
			System.out.println("Error: Failed to build the program.");
			org.jocl.CL.clGetProgramBuildInfo(program, devices[0], org.jocl.CL.CL_PROGRAM_BUILD_LOG, 0, null, logSize);
			byte[] logData = new byte[(int)logSize[0]];
			org.jocl.CL.clGetProgramBuildInfo(program, devices[0], org.jocl.CL.CL_PROGRAM_BUILD_LOG, logSize[0], Pointer.to(logData), null);
			sb.append(new String(logData, 0, logData.length - 1));
			sb.append("\n");
			String buildInfo = sb.toString();

			return buildInfo;
		}
	  
	  private CLUtil() {
	  }
	
}
