#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include<fftw3.h>
#include<omp.h>

#include"nbody.h"

/*------------------------------GLOBAL VARIABLES-------------------------------*/
//-----------------------------------------------------------------------------//
//                              from N-body code 
//-----------------------------------------------------------------------------//

float  vhh, // Hubble parameter in units of 100 km/s/Mpc
  vomegam, // Omega_matter, total matter density (baryons+CDM) parameter
  vomegalam, // Cosmological Constant 
  vomegab, //Omega_baryon
  sigma_8_present, // Last updated value of sigma_8 (Presently PLANCK+WMAP)
  vnn; //Spectral index of primordial Power spectrum

long N1, N2, N3; // box dimension (grid) 
int NF, // Fill every NF grid point 
  Nbin; // Number of bins to calculate final P(k) (output)

float LL; // grid spacing in Mpc

long MM; // Number of particles

int zel_flag=1, // memory allocation for zel is 3 times that for nbody
  fourier_flag; //for fourier transfrom
float  DM_m, // Darm matter mass of simulation particle in (10^10 M_sun h^-1) unit 
  norm, // normalize Pk
  pi=M_PI;

io_header header1;

//-----------------------------------------------------------------------------//
//                          from N-body code  done
//-----------------------------------------------------------------------------//



//-----------------------------------------------------------------------------//
//                           needed for N-bdy funcs 
//-----------------------------------------------------------------------------//

float ***ro; // for density
fftwf_plan p_ro; // for FFT
fftwf_plan q_ro; // for FFT

//-----------------------------------------------------------------------------//
//                           needed for N-bdy funcs done  
//-----------------------------------------------------------------------------//

//----------------------arrays for storing ionization data---------------------//

float ***nh, // stores neutral hydrogen on grid points
  ***nhs,    // stores smoothed neutral hydrogen on grid point
  ***ngamma, // stores photon number on grid points
  ***ngammas, // stores smoothed photon number on grid points
  ***nxion;  // stores ionization fractions for different nions on grid points

/*----------------------------GLOBAL VARIABLES DONE----------------------------*/


void main()
{
  long int seed;
  FILE  *inp,*outpp;
  
  int i;
  long ii,jj, kk,ll, tmp;
  int sfac;
  
  float vaa;  // final scale factor
  
  //-----------------done variables for non-uniform recombination------------//
  
  double *power_P0, *power_P2, *power_P4, *kmode; // arrays for power spectrum 
  double *no;
  
  float dr,r_min,r_max; // radious for smoothing
  
  char file[100], file1[100], file2[100], num[8], num1[8], num2[8];
  
  float nion,xh1; // to store ionization fraction and neutral hydrogen fraction
  double vion, roion; // to store vol. avg. and mass avg. ionization fraction
  
  int output_flag,in_flag;
  
  long totcluster; // total no. of haloes in halo_catalogue
  float robar,Radii;
  double robarhalo; //no. of dark matter paricle density per (grid)^3
  float vfac; //coefficient for redshift space distortion
  
  float **rra,**vva,**data, //to store particle positions and velocities
    **halo;
  
  double t,T=omp_get_wtime(); // for timing
  
  int Noutput;
  float *nz;
  
  
  /*---------------------------------------------------------------------------*/
  /* Read input parameters for the simulation from the file "input.nbody_comp" */
  /*---------------------------------------------------------------------------*/
  
  inp=fopen("input.nbody_comp","r");
  fscanf(inp,"%ld%*d",&tmp);
  fscanf(inp,"%*f%*f%*f%*f");
  fscanf(inp,"%*f%*f");
  fscanf(inp,"%ld%ld%ld%*d%*f",&tmp,&tmp,&tmp);
  fscanf(inp,"%*d%*d");
  fscanf(inp,"%*f%*f");  /* time step, final scale factor*/
  fscanf(inp,"%d",&Noutput);
  
  nz=(float*)calloc(Noutput,sizeof(float)); // array to store Noutput 
  
  for(i=0;i<Noutput;i++)
    fscanf(inp,"%f",&nz[i]);
  
  fclose(inp);
  
  /*-----------------------------------------------------------*/
  
  
  //---------------------------------------------------------------------------//
  //-------------parameters read from input file. Check this ------------------//
  
  sfac=2;
  Nbin=10;
  nion=23.21;
  
  vion=0.0;
  roion=0.0;

  //calculating max and min radius for smoothing in units of grid size
      
  r_min=1.;
  r_max=20.0/LL; // Mpc/LL in grid unit

  //r_max=pow((1.*N1*N2*N3),(1./3.))/2.;
      
  
  //---------------------------------------------------------------------------//
  
  system("mkdir ionz_out");
  
  /*-----------------------------read nbody output-----------------------------*/
  
  for(i=0;i<Noutput;i++)
    {
      strcpy(file,"output.nbody_");
      sprintf(num,"%3.3f",nz[i]);
      strcat(file,num);
      
      read_output(file,1,&seed,&output_flag,&in_flag,rra,vva,&vaa); // only read header
      
      if(i==0)
	{
	  rra = allocate_float_2d(MM,3);
          vva = allocate_float_2d(MM,3);
	  data = allocate_float_2d(MM,5);
	}
      
      read_output(file,2,&seed,&output_flag,&in_flag,rra,vva,&vaa); // read data
      
      printf("ok read nbody output = %e\n",omp_get_wtime()-t);
      
      //---------------------------------------------------------------------------//
      
      //-------------------------reading the halo catalogue------------------------//
      
      t=omp_get_wtime();
      strcpy(file1,"halo_catalogue_");
      sprintf(num1,"%3.3f",nz[i]);
      strcat(file1,num1);
      
      
      read_fof(file1,1,&output_flag,&totcluster,halo,&vaa);
      
      halo = allocate_float_2d(totcluster,7);
      
      read_fof(file1,2,&output_flag,&totcluster,halo,&vaa);
      
      printf("ok read halo catalogue = %e\n",omp_get_wtime()-t);
      
      //---------------------------------------------------------------------------//
      
      //-----------------------------Redefine grid---------------------------------//
      
      
      N1=N1/sfac;  N2=N2/sfac;  N3=N3/sfac;// new grid dimensions 
      LL=LL*sfac; 
      robar=MM/(1.*N1*N2*N3); // mean numbder density (grid)^{-3}
      vfac=1./(Hf(vaa)*vaa*vaa); // for redshift space distortion
      
      //---------------------------------------------------------------------------//
      
      for(ii=0;ii<MM;ii++)
	{
	  data[ii][0] = rra[ii][0]/(1.*sfac);
	  data[ii][1] = rra[ii][1]/(1.*sfac);
	  data[ii][2] = rra[ii][2]/(1.*sfac);
	  
	  data[ii][3] = (rra[ii][2] + vfac*vva[ii][2])/(1.*sfac); // redshift space distortion applied
	  data[ii][3] += N3*1.;
	  data[ii][3] = data[ii][3]-1.0*N3*(int)(floor(data[ii][3])/(1.*N3));
	  
	  data[ii][4] = 1.;  // same mass for all particles
	}
      
      /*----------------------------------------------------------------*/
      
      for(ii=0;ii<totcluster;ii++)
	{
	  halo[ii][1] /= (1.*sfac);
	  halo[ii][2] /= (1.*sfac);
	  halo[ii][3] /= (1.*sfac);
	}
      
      /*----------------------------------------------------------------*/
      
      if(i==0)
	{
	  Setting_Up_Memory_For_ionz();
	  
          /*---------allocate memory for power spectrum and k modes--------*/
	  
          kmode=calloc((size_t)Nbin,sizeof(double));
          power_P0=calloc((size_t)Nbin,sizeof(double));
          power_P2=calloc((size_t)Nbin,sizeof(double));
          power_P4=calloc((size_t)Nbin,sizeof(double)); 
          no=calloc((size_t)Nbin,sizeof(double));
	}
      
      /*----------------------------------------------------------------*/
      
      t=omp_get_wtime();
      
      /* calculating the halo mass density at each grid point */

      MM=totcluster;
      cic_vmass(ngamma, halo, 1, 2, 3, 0);  
      
      /*----------------------------------------------------------------*/
      
      MM=header1.npart[1];
      cic_vmass(nh, data, 0, 1, 2, 4);
      
      printf("ok cic_vmass= %e\n",omp_get_wtime()-t);
      
      /*----------------------------------------------------------------*/
      
      t=omp_get_wtime();
      
      //---------------------subgrid re-ionization----------------------//
      
      t=omp_get_wtime();
      
      for(ii=0;ii<N1;ii++)
	for(jj=0;jj<N2;jj++)
	  for(kk=0;kk<N3;kk++)
	    {
	      if(nh[ii][jj][kk]>nion*ngamma[ii][jj][kk]) // checking ionization condition
		{
		  nxion[ii][jj][kk]=nion*ngamma[ii][jj][kk]/nh[ii][jj][kk];
		}
	      
	      else
		{
		  nxion[ii][jj][kk]=1.;
		}
	    }
      
      //printf("ok sub grid re-ionization = %e\n",omp_get_wtime()-t);
      
      /*----------------------------------------------------------------*/
      
      //-------------calculating avg. ionization fraction---------------//
      
      /*----------------------------------------------------------------*/
      /*----------------------------------------------------------------*/
      
      /*----------------------------------------------------------------*/
      /*                        smoothing                               */
      /*----------------------------------------------------------------*/
      
      t=omp_get_wtime();
      
      Radii=r_min;
      
      while(Radii < r_max)
	{
	  for(ii=0;ii<N1;ii++)
	    for(jj=0;jj<N2;jj++)
	      for(kk=0;kk<N3;kk++)
		{
		  nhs[ii][jj][kk]=nh[ii][jj][kk];
		  ngammas[ii][jj][kk]=ngamma[ii][jj][kk];
		}
	  //printf("starting smoothing for radius of size %e\n",Radii);
	  
	  smooth(nhs,Radii);
	  
	  smooth(ngammas,Radii);
	  
	  
	  for(ii=0;ii<N1;ii++)
	    for(jj=0;jj<N2;jj++)
	      for(kk=0;kk<N3;kk++)
		{
		  if(nhs[ii][jj][kk]<=nion*ngammas[ii][jj][kk])  // checking ionization condition
		    nxion[ii][jj][kk]=1.;
		}
	  
	  dr=(Radii*0.1) < 2.0 ? (Radii*0.1) : 2.0; //increment of the smoothing radius
	  Radii += dr;
	}
      
      printf("ok smoothing = %e\n",omp_get_wtime()-t);
      
      /*----------------------------------------------------------------*/
      
      /*----------------------------------------------------------------*/
      
      
      t=omp_get_wtime();
      
      //---------------calculating avg. neutral fraction-------------*/

      vion =0.0;
      roion=0.0;
      
      /*----------------------------------------------------------------*/
      
      strcpy(file2,"ionz_out/HI_map_");
      sprintf(num2,"%3.3f",nz[i]);
      strcat(file2,num2);
      outpp=fopen(file2,"w");
      
      fwrite(&N1,sizeof(int),1,outpp);
      fwrite(&N2,sizeof(int),1,outpp);
      fwrite(&N3,sizeof(int),1,outpp);

      for(ii=0;ii<N1;ii++)
	for(jj=0;jj<N2;jj++)
	  for(kk=0;kk<N3;kk++)
	    {
  	      xh1=(1.-nxion[ii][jj][kk]);
  	      xh1=(xh1 >0.0)? xh1: 0.0;
	      
  	      nxion[ii][jj][kk]=xh1; // store x_HI instead of x_ion
  	      nhs[ii][jj][kk]=xh1*nh[ii][jj][kk]; // ro_HI on grid
	      
  	      vion+=(double)xh1;
 	      roion+=(double)nhs[ii][jj][kk];
	      
	      fwrite(&nhs[ii][jj][kk],sizeof(float),1,outpp);
	    }
      
      fclose(outpp);
      
      /*----------------------------------------------------------------*/
      
      roion/=(1.*N1*N2*N3); // mean HI density
      
      /*----------------------------------------------------------------*/
      
      calpow_mom(nhs,Nbin,power_P0,kmode,power_P2,power_P4,no); // calculates moments of redshift space power spectrum
      
      
      strcpy(file2,"ionz_out/pk.ionz");
      sprintf(num2,"%4.3f",roion/robar);
      strcat(file2,num2);
      strcat(file2,"_");
      sprintf(num2,"%3.3f",nz[i]);
      strcat(file2,num2);
      
      outpp=fopen(file2,"w");
      
      for(ii=0;ii<Nbin;++ii)
	{
	  power_P0[ii]/=(roion*roion);
	  power_P2[ii]/=(roion*roion);
	  power_P4[ii]/=(roion*roion);
	  
	  fprintf(outpp,"%e %e %e %e %ld\n",kmode[ii],power_P0[ii],power_P2[ii],power_P4[ii],(long)no[ii]);
	}
      
      fclose(outpp);
      
      /*----------------------------------------------------------------*/
      
      vion/=(1.*N1*N2*N3); // volume avg xHI
      roion/=robar; // divide by H density to get mass avg. xHI
      
      printf("vol. avg. x_HI=%e, mass avg. x_HI=%e\n",vion,roion);
      
      
      /*----------------------------------------------------------------*/
      /*            Do the same for redshift space                      */
      /*----------------------------------------------------------------*/
      
      density_2_mass(nxion, data, 0, 1, 2, 4);   // get particles HI masses from HI density
      
      cic_vmass(nhs, data, 0, 1, 3, 4);  // convert particles HI masses to  HI density
      
      //------------calculating avg. ionization fraction----------------//
      
      roion=0.0;
      
      strcpy(file2,"ionz_out/HI_maprs_");
      sprintf(num2,"%3.3f",nz[i]);
      strcat(file2,num2);
      outpp=fopen(file2,"w");
      
      fwrite(&N1,sizeof(int),1,outpp);
      fwrite(&N2,sizeof(int),1,outpp);
      fwrite(&N3,sizeof(int),1,outpp);

      for(ii=0;ii<N1;ii++)
	for(jj=0;jj<N2;jj++)
	  for(kk=0;kk<N3;kk++)
	    {
	      roion+=(double)nhs[ii][jj][kk];
	      fwrite(&nhs[ii][jj][kk],sizeof(float),1,outpp);
	    }
      fclose(outpp);
      
      roion/=(1.*N1*N2*N3); // mean HI density
      
      /*----------------------------------------------------------------*/
      
      calpow_mom(nhs,Nbin,power_P0,kmode,power_P2,power_P4,no); // calculates moments of redshift space power spectrum
      
      strcpy(file2,"ionz_out/pk.ionzs");
      sprintf(num2,"%4.3f",roion/robar);
      strcat(file2,num2);
      strcat(file2,"_");
      sprintf(num2,"%3.3f",nz[i]);
      strcat(file2,num2);
      
      outpp=fopen(file2,"w");
      
      for(ii=0;ii<Nbin;++ii)
	{
	  power_P0[ii]/=(roion*roion);
	  power_P2[ii]/=(roion*roion);
	  power_P4[ii]/=(roion*roion);
	  
	  fprintf(outpp,"%e %e %e %e %ld\n",kmode[ii],power_P0[ii],power_P2[ii],power_P4[ii],(long)no[ii]);
	}
      fclose(outpp);
      
      /*----------------------------------------------------------------*/
      
      roion/=robar; // divide by H density to get mass avg. xHI
      printf("mass avg. x_HI=%e\n",roion);
      
      printf("ok time taken= %e\n\n",omp_get_wtime()-t);
      
      free(halo);
    }

  free(rra);
  free(vva);
  free(data);
  free(ro);
  free(nh);
  free(nhs);
  free(ngamma);
  free(ngammas);
  free(nxion);


    
  printf("done. Total time taken = %dhr %dmin %dsec\n",(int)((omp_get_wtime()-T)/3600), (int)((omp_get_wtime()-T)/60)%60, (int)(omp_get_wtime()-T)%60);
}
