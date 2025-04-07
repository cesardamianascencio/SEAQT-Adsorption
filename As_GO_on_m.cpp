/*
  Modified Replica Exchange Wang–Landau (REWL) Algorithm for Adsorption Simulations  
(c) Adriana Saldana et al., based on original code by Thomas Vogel and Ying Wai Li (2013–2019)  

License: Attribution–ShareAlike 4.0 International (CC BY-SA 4.0)  
https://creativecommons.org/licenses/by-sa/4.0/legalcode  

You are free to:  
- Share – copy and redistribute this material in any medium or format  
- Adapt – remix, transform, and build upon the material for any purpose, even commercially  

(These freedoms cannot be revoked as long as you comply with the license terms.)  

Under the following terms:  

1. Attribution – You must give appropriate credit and must not remove this notice from any file containing parts of this code. You may give credit in any reasonable manner, but not in any way that suggests the original authors endorse you or your use.  
   If you publish data or results obtained using this code or modifications of it, please cite the following original publications:  
   - T. Vogel et al., Phys. Rev. Lett. 110 (2013) 210603  
   - T. Vogel et al., Phys. Rev. E 90 (2014) 023302  
   Additional references include:  
   - T. Vogel et al., J. Phys.: Conf. Ser. 487 (2014) 012001  
   - Y. W. Li et al., J. Phys.: Conf. Ser. 510 (2014) 012012  

   For the modified version presented here, developed to support adsorption simulations within the Steepest-Entropy-Ascent Quantum Thermodynamics (SEAQT) framework, please cite:  
   - Adriana Saldana-Robles, Cesar Damian, William T. Reynolds Jr., Michael R. von Spakovsky, Model for Predicting Adsorption Isotherms and the Kinetics of Adsorption via Steepest-Entropy-Ascent Quantum Thermodynamics, (2025).  

2. ShareAlike – If you modify, transform, or build upon the material, you must distribute your contributions under the same license as the original.

3. No additional restrictions – You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

Notices:  
You do not have to comply with the license for elements of the material that are in the public domain or where your use is permitted by an applicable exception or limitation. No warranties are given. This license may not provide all permissions required for your intended use (e.g., publicity, privacy, or moral rights).

 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <iostream>
#include <float.h>

int switched = INT_MAX;

int iterr; // Apart of an added debug output. Effictively just a counter for the number of repeated histogram values when flatness is checked
int repv = 0; // Holds the value of the (potentially) repeated minvalue of the histogram
int phrepv; // Holds the value of the previous minimum histogram value to compare with the prior variable


const int L1dim = 70;                            // linear dimension
const int Dimension = 2;                        // not sure at this moment if other than 2D is actually implemented
const int numberspins = pow(L1dim, Dimension);  // total number of spins
const int numberneighbors = 4;                  // for 2D square lattice (others not implemented right now)
const int factor = 1;
const int factor1 = 1;
const int factor2 = 1; // for 2D square lattice (others not implemented right now)
//const int Eglobalmin = -48;          // minimum energy for 2D square lattice Potts model
const int Eglobalmin = 55000;
const int Eglobalmax = 75000;// -188/2;                       // maximum energy of Potts model
const int nS0 = 2;
const int nS1 = 3;
const int nS3 = 1;
const int nS2 = nS1;
const int nS4 = numberspins - nS0 -nS1 - nS2 - nS3;

//int matrix1[5][5]= {{229, -51, -52, -98, 49}, {-51, 11, 11, 22, -11}, {-52, 11, 0, 22, -11}, {-98, 22, 22, 42, -21}, {49, -11, -11, -21, 11}};


//int matrix2[5][5]= {{162, -36, -37, -69, 35}, {-36, 8, 8, 15, -8}, {-37, 8, 0, 16, -8}, {-69, 15, 16, 29, -15}, {35, -8, -8, -15, 7}};


//int matrix3[5][5]= {{114, -25, -26, -49, 24}, {-25, 6, 6, 11, -5}, {-26, 6, 0, 11, -6}, {-49, 11, 11, 21, -11}, {25, -5, -6, -11, 5}};

int matrix1[5][5] = {{23, -5, -5, -10, 5}, {-5, 1, 1, 2, -1}, {-5, 1, 0, 2, -1}, {-10, 2, 2, 4, -2}, {5, -1, -1, -2, 1}};

int matrix2[5][5] = {{16, -4, -4, -7, 3}, {-4, 1, 1, 2, -1}, {-4, 1, 0, 2, -1}, {-7, 2, 2, 3, -1}, {3, -1, -1, -1, 1}};

int matrix3[5][5] = {{11, -3, -3, -5, 2}, {-3, 1, 1, 1, -1}, {-3, 1, 0, 1, -1}, {-5, 1, 1, 2, -1}, {2, -1, -1, -1, 1}};







int Eglobalwidth=abs(Eglobalmin-Eglobalmax);
const int bctype = 0;                           // type of boundary condition: 0 - periodic; 1 - Braskamp Kunz
int q = 5;                               // number of different possible spin states (q-state Potts model)

int* latticepoint;                      // list containing values of all spins
int* neighbor;                          // list containing indices of neighbors for all spins
int* neighbor2;                          // list containing indices of second neighbors for all spins
int* neighbor3;                          // list containing indices of diagonal neighbors for all spins
 
double* HE;                             // energy histogram
double* lngE;                           // ln g(E) estimator
double* lngE_buf;                       // ln g(E) estimator buffer for exchange
double* pseudolngE;
double* real_lngE;
double* real_lngE_buf;
double* microT;
double* microT_buf;
int lowest_energy = -1*Eglobalmin; // Initialize with maximum possible integer value
int* lowest_energy_lattice;  // Pointer to store the lattice configuration


int hist_size = (-Eglobalmin+Eglobalmax) + 1;        // histogram size


int rseed;                              // seed for random number generator
int energy, localenergy;

double Emin, Emax;                       // doubles as calculation of boundaries in Energy is not in 'int'
int Eminindex, Emaxindex, Estartindex;  // local boundaries as index 

// MPI; set up with local communicators for replica exchange (RE)
// Needed so that two processes can talk w/o depending on / bothering the others
int numprocs, myid, multiple, comm_id;
// each process belongs to two local groups,
// one to communicate to left neighbor and one to communicate to right neighbor
// each process has different loca IDs in different communicatore, in general 
int mylocalid[2];                       // id in local communicators 
MPI_Comm* mpi_local_comm;
MPI_Group* mpi_local_group;
MPI_Group world;
MPI_Status status;
int merge_hists = 1;                    // flag whether or not to merge histograms between iterations for multiple walkers on the same energy range

// to keep track of exchange statistics
int tryleft, tryright, exchangeleft, exchangeright;

// File handlers for I/O
FILE* file;
FILE* stdoutlog;
FILE* wanderlog;
FILE* adslatlog;
char filename[50];
char stdoutlogname[128];
char wanderlogname[128];
char adslat[128];

int ret_status;

double flatratio;
double flatmin;

// to track execution times
time_t timenow, timestart, timeend;


int check_spin_counts(int* latticepoint, int numberspins, int nS0, int nS1, int nS2, int nS3, int nS4) 
{
    int counts[5] = {0, 0, 0, 0, 0};  // Assuming spins can only have values 0 to 4
    for(int i = 0; i < numberspins; i++)
    {
        counts[latticepoint[i]]++;
    }
    // Check if the counts match the expected numbers
    if (counts[0] == nS0 && counts[1] == nS1 && counts[2] == nS2 && counts[3] == nS3 && counts[4] == nS4)
    {
        return 1;  // The spin counts match the expected values
    } else 
    {
        return 0;  // The spin counts do not match
    }
}


int check_neighbor_spins(int* latticepoint, int* neighbor, int numberspins)
{
    for (int i = 0; i < numberspins; i++)
    {
        int count_neighbor = 0;
        if (latticepoint[i] == 0) // Check only for spin 0
        {
            // Check the four neighbors: above, right, below, left
            for (int j = 0; j < 4; j++)
            {
                
                int neighborIndex = neighbor[4*i + j]; // Get the index of the j-th neighbor
                int neighborIndex2 = neighbor2[4*i + j];
                int neighborIndex3 = neighbor3[4*i + j];

                // Check for valid neighbor index and avoid self-reference
                if (neighborIndex >= 0 && neighborIndex < numberspins && neighborIndex != i)
                {
                    if ( latticepoint[neighborIndex] == 1 )
                    {
                        count_neighbor++;
                        return 1; // Found at least one neighbor being 1 or 2 near a spin 0
                    }
                }
            }
        }
    }

    return 0; // No spin 0 has a neighbor of 1 or 2
}



int count_spin_1_neighbors_of_spin_0(int* latticepoint, int* neighbor, int numberspins)
{
    int total_count = 0;

    for (int i = 0; i < numberspins; i++)
    {
        if (latticepoint[i] == 0) // Check only for spin 0
        {
            int count = 0; // Count of spin 1 neighbors for this spin 0

            // Check the four neighbors: above, right, below, left
            for (int j = 0; j < 4; j++)
            {
                int neighborIndex = neighbor[4*i + j]; // Get the index of the j-th neighbor
                int neighborIndex2 = neighbor2[4*i + j];
                int neighborIndex3 = neighbor3[4*i + j];

                // Check for valid neighbor index and avoid self-reference
                if (neighborIndex >= 0 && neighborIndex < numberspins && neighborIndex != i)
                {
                    if (latticepoint[neighborIndex] == 1 || latticepoint[neighborIndex] == 2 || 
                        latticepoint[neighborIndex2] == 1 || latticepoint[neighborIndex2] == 2 ||
                        latticepoint[neighborIndex3] == 1 || latticepoint[neighborIndex3] == 2)
                    {
                        //count++; // Found a neighbor being 1 or 2 near a spin 0
                        count=1;
                    }
                }
            }

            total_count += count; // Add the count for this spin to the total
        }
    }

    return total_count; // Return the total count of spin 1 neighbors for all spin 0
}


void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}


void keypressed()     // just for developing / manual debugging
{
	for (;;)
		if (getchar() == 27) break;
}



void init_neighbors() // create neighbor list
{
  // neighbor contains the index of the neighboring spin
  // for each spin there are four neighbors in this order: above, right, below, left
  neighbor = (int*) malloc(numberspins * numberneighbors * sizeof(int));
  neighbor2 = (int*) malloc(numberspins * numberneighbors * sizeof(int));
  neighbor3 = (int*) malloc(numberspins * numberneighbors * sizeof(int));

    
  for (int i=0; i<numberspins; i++)    // in general
    {
      neighbor[4*i]   = i - L1dim;     // above
      neighbor[4*i+1] = i + 1;         // right
      neighbor[4*i+2] = i + L1dim;     // below
      neighbor[4*i+3] = i - 1;         // left
      // Second neighbors
      neighbor2[4*i]  = i - 2*L1dim;   // above-above
      neighbor2[4*i+1]  = i + 2;       // right-right
      neighbor2[4*i+2]  =i + 2*L1dim;       // below-below
      neighbor2[4*i+3]  = i -2;       // left-left
      // Diagonal neighbors
      neighbor3[4*i]  = i - L1dim-1;   // lower left
      neighbor3[4*i+1]  = i - L1dim+1;       // lower right
      neighbor3[4*i+2]  =i + L1dim-1;       // upper left
      neighbor3[4*i+3]  = i + L1dim+1;       // upper right
          
        
    };

  if (bctype == 0)   // periodic BC
    for (int i=0; i<numberspins; i++)                  // now treat boundaries separately
    {
        if (i < L1dim)                                  // bottom row
        {
            neighbor[4*i] = numberspins - L1dim + i;
            neighbor2[4*i] = numberspins - 2*L1dim + i;
            neighbor3[4*i] = numberspins - L1dim + i-1;
            neighbor3[4*i+1] = numberspins - L1dim + i+1;
            
        }
        if ( (i >= L1dim) &&  (i < 2*L1dim) )          // inner bottom row
        {
            neighbor2[4*i] = numberspins - 2*L1dim + i;
            
        }
        if ((i+1)%L1dim == 0)                          // rightmost column
        {
            neighbor[4*i+1] = i + 1 - L1dim;
          
            if(i<numberspins)
            {
                neighbor2[4*i+1] = i + 2 -L1dim;
            }
            
            if(i==(L1dim-1))
           {
               neighbor3[4*i+1] = numberspins-L1dim;
               neighbor3[4*i+3] = i+1;
           }
            if(i>(L1dim-1))
            {
                neighbor3[4*i+1] = i-2*L1dim+1;
                neighbor3[4*i+3] = i+1;
            }
        }
        if ((i+2)%L1dim == 0)                          // inner rightmost column
        {
            neighbor2[4*i+1] = i + 2 -L1dim;
        }
        if (i > (numberspins - L1dim - 1) )            // top row
        {
            neighbor[4*i+2] = i - (numberspins - L1dim);
            neighbor2[4*i+2] = i - (numberspins - 2*L1dim);
            if(i==numberspins-1)
            {
                neighbor3[4*i+2] =L1dim-2;
                neighbor3[4*i+3] =0;
            }
            if(i!=numberspins-1)
            {
                neighbor3[4*i+2] =i+L1dim-numberspins-1;
                neighbor3[4*i+3] =i+L1dim-numberspins+1;
            }
        }
        if ( (i > (numberspins - 2*L1dim - 1)) && (i <= (numberspins - L1dim - 1)) )            // inner top row
        {
            neighbor2[4*i+2] = i - (numberspins - 2*L1dim);
        
        }
        if (i%L1dim == 0)                              // leftmost column
        {
            neighbor[4*i+3]  = i - 1 + L1dim;
            neighbor2[4*i+3] = i - 2 + L1dim;
            if(i==0) 
            {
                neighbor3[4*i] = numberspins-1;
                neighbor3[4*i+2] = i+ 2*L1dim-1;
            }
            if(i>0)
            {
                neighbor3[4*i] = i-1;
                neighbor3[4*i+2] = i+ 2*L1dim-1;
            }
        }
        if ((i-1)%L1dim == 0)                              // inner leftmost column
        {
            neighbor2[4*i+3] = i - 2 + L1dim;
        }
        if (i==numberspins-L1dim)
        {
            neighbor3[4*i+2] = 1;
            neighbor3[4*i+3] = L1dim-1;
        }
    };
    
    /*printf("\n --------------------00 -- 00------------------- \n");
    //print neighbor list (for manual debugging);
    for (int i =0; i< L1dim*L1dim; i++)
    {
        printf(" %4d : ( %4d, %4d, %4d, %4d ) ",i,neighbor[4*i],neighbor[4*i+1],neighbor[4*i+2],neighbor[4*i+3]);
        printf("\n %4d : ( %4d, %4d, %4d, %4d ) ",i,neighbor2[4*i],neighbor2[4*i+1],neighbor2[4*i+2],neighbor2[4*i+3]);
        printf("\n %4d : ( %4d, %4d, %4d, %4d ) ",i,neighbor3[4*i],neighbor3[4*i+1],neighbor3[4*i+2],neighbor3[4*i+3]);
        printf("\n --------------------------------------- \n");
    };*/

  if (bctype == 1) // Braskamp Kunz BC
     
    for (int i=0; i<numberspins; i++)                  // now treat boundaries separately
      {
    if ((i+1)%L1dim == 0)                          // rightmost column
      neighbor[4*i+1] = i + 1 - L1dim;
    if (i%L1dim == 0)                              // leftmost column
      neighbor[4*i+3] = i - 1 + L1dim;
    if (i < L1dim)                                 // top row
      neighbor[4*i] = numberspins;
    if (i > (numberspins - L1dim - 1) )            // bottom row
      neighbor[4*i+2] = numberspins + (i%2);
      };
    
}



// Our case ---------------------------------------------------------------------------------
int totalenergy()         // returns total energy of system
{
    int e = 0;
    
    for (int i = 0; i < numberspins; i++)
    {
        int holder =0;
        int r=(latticepoint[i]);
        //if(r==0) r=5;
    
        int rr, rr2,rrd;
        if (bctype == 0)    // periodic boundaries
            for (int j = 0; j < 4; j++)
            {
                rr=(latticepoint[neighbor[4 * i + j]]);
                rr2=(latticepoint[neighbor2[4 * i + j]]);
                rrd=(latticepoint[neighbor3[4 * i + j]]);
//Neighbor 1 (d1=3.21)-----------------------------------------------------
                if(r==0 && rr==0) e+=factor*matrix1[0][0];
                if(r==0 && rr==1) e+=factor*matrix1[0][1];
                if(r==0 && rr==2) e+=factor*matrix1[0][2];
                if(r==0 && rr==3) e+=factor*matrix1[0][3];
                if(r==0 && rr==4) e+=factor*matrix1[0][4];

                if(r==1 && rr==0) e+=factor*matrix1[1][0];
                if(r==1 && rr==1) e+=factor*matrix1[1][1];
                if(r==1 && rr==2) e+=factor*matrix1[1][2];
                if(r==1 && rr==3) e+=factor*matrix1[1][3];
                if(r==1 && rr==4) e+=factor*matrix1[1][4];

                if(r==2 && rr==0) e+=factor*matrix1[2][0];
                if(r==2 && rr==1) e+=factor*matrix1[2][1];
                if(r==2 && rr==2) e+=factor*matrix1[2][2];
                if(r==2 && rr==3) e+=factor*matrix1[2][3];
                if(r==2 && rr==4) e+=factor*matrix1[2][4];

                if(r==3 && rr==0) e+=factor*matrix1[3][0];
                if(r==3 && rr==1) e+=factor*matrix1[3][1];
                if(r==3 && rr==2) e+=factor*matrix1[3][2];
                if(r==3 && rr==3) e+=factor*matrix1[3][3];
                if(r==3 && rr==4) e+=factor*matrix1[3][4];

                if(r==4 && rr==0) e+=factor*matrix1[4][0];
                if(r==4 && rr==1) e+=factor*matrix1[4][1];
                if(r==4 && rr==2) e+=factor*matrix1[4][2];
                if(r==4 && rr==3) e+=factor*matrix1[4][3];
                if(r==4 && rr==4) e+=factor*matrix1[4][4];
                   
                                
//Neighbor 2 (d2=4.53 diagonal)----------------------------------------------

                if(r==0 && rrd==0) e+=factor1*matrix2[0][0];
                if(r==0 && rrd==1) e+=factor1*matrix2[0][1];
                if(r==0 && rrd==2) e+=factor1*matrix2[0][2];
                if(r==0 && rrd==3) e+=factor1*matrix2[0][3];
                if(r==0 && rrd==4) e+=factor1*matrix2[0][4];

                if(r==1 && rrd==0) e+=factor1*matrix2[1][0];
                if(r==1 && rrd==1) e+=factor1*matrix2[1][1];
                if(r==1 && rrd==2) e+=factor1*matrix2[1][2];
                if(r==1 && rrd==3) e+=factor1*matrix2[1][3];
                if(r==1 && rrd==4) e+=factor1*matrix2[1][4];

                if(r==2 && rrd==0) e+=factor1*matrix2[2][0];
                if(r==2 && rrd==1) e+=factor1*matrix2[2][1];
                if(r==2 && rrd==2) e+=factor1*matrix2[2][2];
                if(r==2 && rrd==3) e+=factor1*matrix2[2][3];
                if(r==2 && rrd==4) e+=factor1*matrix2[2][4];

                if(r==3 && rrd==0) e+=factor1*matrix2[3][0];
                if(r==3 && rrd==1) e+=factor1*matrix2[3][1];
                if(r==3 && rrd==2) e+=factor1*matrix2[3][2];
                if(r==3 && rrd==3) e+=factor1*matrix2[3][3];
                if(r==3 && rrd==4) e+=factor1*matrix2[3][4];

                if(r==4 && rrd==0) e+=factor1*matrix2[4][0];
                if(r==4 && rrd==1) e+=factor1*matrix2[4][1];
                if(r==4 && rrd==2) e+=factor1*matrix2[4][2];
                if(r==4 && rrd==3) e+=factor1*matrix2[4][3];
                if(r==4 && rrd==4) e+=factor1*matrix2[4][4];

//Neighbor 3 (d3=5.45)-----------------------------------------------------

                if(r==0 && rr2==0) e+=factor2*matrix3[0][0];
                if(r==0 && rr2==1) e+=factor2*matrix3[0][1];
                if(r==0 && rr2==2) e+=factor2*matrix3[0][2];
                if(r==0 && rr2==3) e+=factor2*matrix3[0][3];
                if(r==0 && rr2==4) e+=factor2*matrix3[0][4];

                if(r==1 && rr2==0) e+=factor2*matrix3[1][0];
                if(r==1 && rr2==1) e+=factor2*matrix3[1][1];
                if(r==1 && rr2==2) e+=factor2*matrix3[1][2];
                if(r==1 && rr2==3) e+=factor2*matrix3[1][3];
                if(r==1 && rr2==4) e+=factor2*matrix3[1][4];

                if(r==2 && rr2==0) e+=factor2*matrix3[2][0];
                if(r==2 && rr2==1) e+=factor2*matrix3[2][1];
                if(r==2 && rr2==2) e+=factor2*matrix3[2][2];
                if(r==2 && rr2==3) e+=factor2*matrix3[2][3];
                if(r==2 && rr2==4) e+=factor2*matrix3[2][4];

                if(r==3 && rr2==0) e+=factor2*matrix3[3][0];
                if(r==3 && rr2==1) e+=factor2*matrix3[3][1];
                if(r==3 && rr2==2) e+=factor2*matrix3[3][2];
                if(r==3 && rr2==3) e+=factor2*matrix3[3][3];
                if(r==3 && rr2==4) e+=factor2*matrix3[3][4];

                if(r==4 && rr2==0) e+=factor2*matrix3[4][0];
                if(r==4 && rr2==1) e+=factor2*matrix3[4][1];
                if(r==4 && rr2==2) e+=factor2*matrix3[4][2];
                if(r==4 && rr2==3) e+=factor2*matrix3[4][3];
                if(r==4 && rr2==4) e+=factor2*matrix3[4][4];

                //holder+=rr;
                //e-=getJValue(r);
            }
        if (bctype == 1)    // Braskamp Kunz boundaries
            for (int j = 1; j < 3; j++)
            {
                if (latticepoint[i] == latticepoint[neighbor[4 * i + j]])
                    e--;
            }
        //e+=r*holder/2;
    };

    if (bctype == 1)        // Braskamp Kunz boundaries, add fixed upper boundaries
        for (int i = 0; i < L1dim; i++)
        {
            if (latticepoint[i] == latticepoint[neighbor[4 * i]])
                e--;
        }
    
    /*printf(" e = %d \n",e);*/
    return (e);
}



int local_energy(int i)   // returns energy of a single spin
{
    double eloc = 0;

    int holder =0;
    int r=(latticepoint[i]);
    
    
    //if(r==0) r=5;
    int rr, rr2, rrd;
    if (bctype == 0)    // periodic boundaries
        for (int j = 0; j < 4; j++)
        {
            rr=(latticepoint[neighbor[4 * i + j]]);
            rr2=(latticepoint[neighbor2[4 * i + j]]);
            rrd=(latticepoint[neighbor3[4 * i + j]]);
            //Nelocighbor 1 (d1=3.21)-----------------------------------------------------
            if(r==0 && rr==0) eloc+=factor*matrix1[0][0];
            if(r==0 && rr==1) eloc+=factor*matrix1[0][1];
            if(r==0 && rr==2) eloc+=factor*matrix1[0][2];
            if(r==0 && rr==3) eloc+=factor*matrix1[0][3];
            if(r==0 && rr==4) eloc+=factor*matrix1[0][4];

            if(r==1 && rr==0) eloc+=factor*matrix1[1][0];
            if(r==1 && rr==1) eloc+=factor*matrix1[1][1];
            if(r==1 && rr==2) eloc+=factor*matrix1[1][2];
            if(r==1 && rr==3) eloc+=factor*matrix1[1][3];
            if(r==1 && rr==4) eloc+=factor*matrix1[1][4];

            if(r==2 && rr==0) eloc+=factor*matrix1[2][0];
            if(r==2 && rr==1) eloc+=factor*matrix1[2][1];
            if(r==2 && rr==2) eloc+=factor*matrix1[2][2];
            if(r==2 && rr==3) eloc+=factor*matrix1[2][3];
            if(r==2 && rr==4) eloc+=factor*matrix1[2][4];

            if(r==3 && rr==0) eloc+=factor*matrix1[3][0];
            if(r==3 && rr==1) eloc+=factor*matrix1[3][1];
            if(r==3 && rr==2) eloc+=factor*matrix1[3][2];
            if(r==3 && rr==3) eloc+=factor*matrix1[3][3];
            if(r==3 && rr==4) eloc+=factor*matrix1[3][4];

            if(r==4 && rr==0) eloc+=factor*matrix1[4][0];
            if(r==4 && rr==1) eloc+=factor*matrix1[4][1];
            if(r==4 && rr==2) eloc+=factor*matrix1[4][2];
            if(r==4 && rr==3) eloc+=factor*matrix1[4][3];
            if(r==4 && rr==4) eloc+=factor*matrix1[4][4];
                               
                                            
            //Nelocighbor 2 (d2=4.53 diagonal)----------------------------------------------

            if(r==0 && rrd==0) eloc+=factor1*matrix2[0][0];
            if(r==0 && rrd==1) eloc+=factor1*matrix2[0][1];
            if(r==0 && rrd==2) eloc+=factor1*matrix2[0][2];
            if(r==0 && rrd==3) eloc+=factor1*matrix2[0][3];
            if(r==0 && rrd==4) eloc+=factor1*matrix2[0][4];

            if(r==1 && rrd==0) eloc+=factor1*matrix2[1][0];
            if(r==1 && rrd==1) eloc+=factor1*matrix2[1][1];
            if(r==1 && rrd==2) eloc+=factor1*matrix2[1][2];
            if(r==1 && rrd==3) eloc+=factor1*matrix2[1][3];
            if(r==1 && rrd==4) eloc+=factor1*matrix2[1][4];

            if(r==2 && rrd==0) eloc+=factor1*matrix2[2][0];
            if(r==2 && rrd==1) eloc+=factor1*matrix2[2][1];
            if(r==2 && rrd==2) eloc+=factor1*matrix2[2][2];
            if(r==2 && rrd==3) eloc+=factor1*matrix2[2][3];
            if(r==2 && rrd==4) eloc+=factor1*matrix2[2][4];

            if(r==3 && rrd==0) eloc+=factor1*matrix2[3][0];
            if(r==3 && rrd==1) eloc+=factor1*matrix2[3][1];
            if(r==3 && rrd==2) eloc+=factor1*matrix2[3][2];
            if(r==3 && rrd==3) eloc+=factor1*matrix2[3][3];
            if(r==3 && rrd==4) eloc+=factor1*matrix2[3][4];

            if(r==4 && rrd==0) eloc+=factor1*matrix2[4][0];
            if(r==4 && rrd==1) eloc+=factor1*matrix2[4][1];
            if(r==4 && rrd==2) eloc+=factor1*matrix2[4][2];
            if(r==4 && rrd==3) eloc+=factor1*matrix2[4][3];
            if(r==4 && rrd==4) eloc+=factor1*matrix2[4][4];

            //Nelocighbor 3 (d3=5.45)-----------------------------------------------------

            if(r==0 && rr2==0) eloc+=factor2*matrix3[0][0];
            if(r==0 && rr2==1) eloc+=factor2*matrix3[0][1];
            if(r==0 && rr2==2) eloc+=factor2*matrix3[0][2];
            if(r==0 && rr2==3) eloc+=factor2*matrix3[0][3];
            if(r==0 && rr2==4) eloc+=factor2*matrix3[0][4];

            if(r==1 && rr2==0) eloc+=factor2*matrix3[1][0];
            if(r==1 && rr2==1) eloc+=factor2*matrix3[1][1];
            if(r==1 && rr2==2) eloc+=factor2*matrix3[1][2];
            if(r==1 && rr2==3) eloc+=factor2*matrix3[1][3];
            if(r==1 && rr2==4) eloc+=factor2*matrix3[1][4];

            if(r==2 && rr2==0) eloc+=factor2*matrix3[2][0];
            if(r==2 && rr2==1) eloc+=factor2*matrix3[2][1];
            if(r==2 && rr2==2) eloc+=factor2*matrix3[2][2];
            if(r==2 && rr2==3) eloc+=factor2*matrix3[2][3];
            if(r==2 && rr2==4) eloc+=factor2*matrix3[2][4];

            if(r==3 && rr2==0) eloc+=factor2*matrix3[3][0];
            if(r==3 && rr2==1) eloc+=factor2*matrix3[3][1];
            if(r==3 && rr2==2) eloc+=factor2*matrix3[3][2];
            if(r==3 && rr2==3) eloc+=factor2*matrix3[3][3];
            if(r==3 && rr2==4) eloc+=factor2*matrix3[3][4];

            if(r==4 && rr2==0) eloc+=factor2*matrix3[4][0];
            if(r==4 && rr2==1) eloc+=factor2*matrix3[4][1];
            if(r==4 && rr2==2) eloc+=factor2*matrix3[4][2];
            if(r==4 && rr2==3) eloc+=factor2*matrix3[4][3];
            if(r==4 && rr2==4) eloc+=factor2*matrix3[4][4];
            
            holder+=rr;
        }
    
 /* for (int i = 0; i < numberspins; i++)
    {   if(i==0) printf("{ {");
        printf("%4i ,", latticepoint[i]);
        if(i==numberspins-1) printf("}");
        if ((i + 1) % L1dim == 0)
            printf("}, \n");
    }
    printf("\n \n");*/
    return (eloc);
}


int propose_update(int w, int *swap_with_backup) {
    // Get the type of spin at position w
    int spin_type_w = latticepoint[w];
    int swap_with;
    int attempts = 0;

    // Find a spin to swap with
    do {
        swap_with = rand() % numberspins; // Randomly select another spin
        attempts++;
        if (attempts > numberspins * 2) {
            // Return 0 to indicate that we couldn't find a spin to swap with
            // Also set swap_with_backup to -1 to indicate no valid swap was found
            *swap_with_backup = -1;
            return 0;
        }
    } while (swap_with == w || latticepoint[swap_with] == spin_type_w); // Make sure it's a different spin

    // At this point, we found a valid spin to swap with
    // Record the index for potential reverting
    *swap_with_backup = swap_with;

    // Calculate the local energy before swapping
    int elocvor_w = local_energy(w);
    int elocvor_swap_with = local_energy(swap_with);

    // Swap the spins
    int temp = latticepoint[w];
    latticepoint[w] = latticepoint[swap_with];
    latticepoint[swap_with] = temp;

    // Calculate the local energy after swapping
    int elocnach_w = local_energy(w);
    int elocnach_swap_with = local_energy(swap_with);

    // Calculate the energy difference due to the swap
    int delta_e = (elocnach_w - elocvor_w) + (elocnach_swap_with - elocvor_swap_with);

    // Return the energy difference, do not swap back here
    //printf("delta_e = %4d \n",delta_e);
    return delta_e;
}



// Our case ---------------------------------------------------------------------------------

int histflat(int imin, int imax, double ratio)
{
	// checks flatness of histograms for _all_ walkers in energy window
	int myflat, otherflat;
	int flatproc, ioffset;
	int flatprocr, ioffsetr;
	int flatness_crit = 1;

	int multimyflat = 0; // bool value of wheter in of the concurrent walkers in a window are flat

	int merge_crit = 0; // merge_hist criteria

	// check own flatness first
	if (flatness_crit == 0)       // Zhou criterion
	{
		// NOT AVAILABLE YET!
		// TODO: CALCULATE histmin FOR ZHOU CRITERION
		//    myflat=1;
		//    for (int x=imin;x<=imax;x++)
		//    if (HE[x]<histmin) myflat=0;
	}
	else if (flatness_crit == 1)  // "Original" percentage criterion
	{
		myflat = 1;
		double minval;
		minval = HE[imin];        // take GS hits as reference


		/*if (myid == 0)
		{
			minval = HE[10];
		}*/

		minval = DBL_MAX; // Ridiculously large arbitrary value
		for (int x = imin; x <= imax; x++)
		{
			if ((lngE[x] > 0 || HE[x] > 0) && HE[x] <= minval)
			{
				minval = HE[x];
			}
		}

		double average = 0.0;
		double count = 0.0;

		for (int x = imin; x <= imax; x++)
		{
			/*if (((x > 5) || (x == 4)) && (HE[x] < minval))
				minval = HE[x];*/
				//(I am not sure right now why I included the first condition ...)

			if (lngE[x] > 0 || HE[x] > 0)
			{
				average += HE[x];
				count++;
			}
		}
		average /= count;

		flatratio = (ratio * average);
		flatmin = (minval);

		if (minval < (ratio * average))
			myflat = 0;
	}

	// now talk to all the other walkers in the energy window
	// (! this whole thing can be reduced to an MPI_Allreduce once there 
	// are separate communicators for energy windows !)

	if (multiple > 1)
	{
		if (merge_crit == 0 && merge_hists == 1)                   // check flatness of other walkers in window
		{
			if (myid % multiple == 0)             // 'root' in energy window, receive individual flatnesses
			{
				for (int i = 1; i < multiple; i++)
				{
					MPI_Recv(&otherflat, 1, MPI_INT, myid + i, 66, MPI_COMM_WORLD, &status);
					myflat *= otherflat;        // all have to be '1' to give '1' in the end (manual '&&')
				}
				for (int i = 1; i < multiple; i++)  // and let everybody know
				{
					MPI_Send(&myflat, 1, MPI_INT, myid + i, 88, MPI_COMM_WORLD);
				}
			}
			else                                // send individual flatness and receive 'merged' flatness
			{
				MPI_Send(&myflat, 1, MPI_INT, myid - (myid % multiple), 66, MPI_COMM_WORLD);
				MPI_Recv(&otherflat, 1, MPI_INT, myid - (myid % multiple), 88, MPI_COMM_WORLD, &status);
				myflat = otherflat;             // replace individual flatness by merged
			}
		}

		if (merge_crit == 1 && merge_hists == 1)                   // check flatness of other walkers in window
		{
			if (myid % multiple == 0)             // 'root' in energy window, receive individual flatnesses
			{
				if (myflat == 1) //Main node multimyflat (checks for the flat process in an energy window
				{
					flatproc = myid; // id of flat process
					ioffset = (myid - (myid % multiple)) - myid; // offset from the main energy window node
					multimyflat = 1;
				}

				for (int i = 1; i < multiple; i++)
				{
					MPI_Recv(&otherflat, 1, MPI_INT, myid + i, 66, MPI_COMM_WORLD, &status);

					if (otherflat == 1) //sets the value of the two variable based on information recieved from the process each one is communicating with
					{
						flatproc = (myid + i);
						ioffset = (myid - (myid % multiple)) - myid;
						multimyflat = 1;
					}

					myflat *= otherflat;        // all have to be '1' to give '1' in the end (manual '&&')
				}
				for (int i = 1; i < multiple; i++)  // and let everybody know
				{
					MPI_Send(&myflat, 1, MPI_INT, myid + i, 88, MPI_COMM_WORLD);
					MPI_Send(&multimyflat, 1, MPI_INT, myid + i, 86, MPI_COMM_WORLD);
					if (multimyflat == 1) // if multimyflat is found to be equal to one flatproc and ioffset 
					{
						MPI_Send(&flatproc, 1, MPI_INT, myid + i, 90, MPI_COMM_WORLD);
						MPI_Send(&ioffset, 1, MPI_INT, myid + i, 92, MPI_COMM_WORLD);
					}
				}
			}
			else                                // send individual flatness and receive merged status variables
			{
				MPI_Send(&myflat, 1, MPI_INT, myid - (myid % multiple), 66, MPI_COMM_WORLD);
				MPI_Recv(&otherflat, 1, MPI_INT, myid - (myid % multiple), 88, MPI_COMM_WORLD, &status);
				MPI_Recv(&multimyflat, 1, MPI_INT, myid - (myid % multiple), 86, MPI_COMM_WORLD, &status);
				if (multimyflat == 1)
				{
					MPI_Recv(&flatprocr, 1, MPI_INT, myid - (myid % multiple), 90, MPI_COMM_WORLD, &status);
					flatproc = flatprocr;
					MPI_Recv(&ioffsetr, 1, MPI_INT, myid - (myid % multiple), 92, MPI_COMM_WORLD, &status);
					ioffset = ioffsetr;
				}
				myflat = otherflat;             // replace individual flatness by merged
			}


			if (multimyflat == 1)
			{
				if (myid != flatproc) // non flat process recieve merged flat process density of states and the myflat status flagged is called to perform the lnge merging procedures in the wl routine
				{
					stdoutlog = fopen(stdoutlogname, "a");
					fprintf(stdoutlog, "Proc %3i: recievied flat DOS from Proc: %3i \t offset: %3i \t multimyflat: %3i\n", myid, flatproc, ioffset, multimyflat);
					fclose(stdoutlog);
					MPI_Recv(&lngE_buf[0], hist_size, MPI_DOUBLE, flatproc, 77, MPI_COMM_WORLD, &status);
					for (int j = 0; j < hist_size; j++) lngE[j] = lngE_buf[j]; // overrides density of states of non flat processes
					myflat = 1;
				}
				else // flat process sends out its density of states values
				{
					for (int i = 0; i < multiple; i++)
					{
						if (myid == flatproc && myid - (myid % multiple) + i != flatproc)
						{
							stdoutlog = fopen(stdoutlogname, "a");
							fprintf(stdoutlog, "Proc %3i: sent flat DOS from Proc: %3i \t offset: %3i \t multimyflat: %3i\n", myid, myid - (myid % multiple) + i, ioffset, multimyflat);
							fclose(stdoutlog);
							MPI_Send(&lngE[0], hist_size, MPI_DOUBLE, myid - (myid % multiple) + i, 77, MPI_COMM_WORLD);
							myflat = 1;
						}
					}
				}
			}
		}
	}
	return (myflat);
	// note: by now, myflat refers to the 'collective' flatness in the energy window,
	// not the flatness of an individual walker
}


void init_lattice(double emin, double emax) // Changes made to bctype check and made latticepoint[numbersign equal to 0
{
	int e, r;

	latticepoint = (int*)malloc((numberspins + 2 + 1) * sizeof(int));
	// Note: we reserve space for 3 extra 'spins':
	// 2 extra to store fix values used for certain boundary conditions
	// 1 extra to carry an replica-id flag

	// find a fast way to create valid initial configurations
	// start at either Eglobalmin or Eglobalmax and change spins randomly
	// until energy is in the middle third of local energy range
	// global maximum of g(E) id at E/N=-0.2 

	//initialize lattice
    // Fill lattice sequentially with the right number of each spin type
    // Seed the random number generator
       srand((unsigned int)time(NULL));

       if ((emin + (emax - emin) / 2) > -0.2 * numberspins) {
           for (int i = 0; i < numberspins; i++) {
               if (i < nS0) latticepoint[i] = 0;
               else if (i < nS0 + nS1) latticepoint[i] = 1;
               else if (i < nS0 + nS1 + nS2) latticepoint[i] = 2;
               else if (i < nS0 + nS1 + nS2 + nS3) latticepoint[i] = 3;
               else latticepoint[i] = 4;
           }
       } else {
           // Initialize the latticepoint with the required number of each state
           int index = 0;
           for(int i = 0; i < nS0; i++) latticepoint[index++] = 0;
           for(int i = 0; i < nS1; i++) latticepoint[index++] = 1;
           for(int i = 0; i < nS2; i++) latticepoint[index++] = 2;
           for(int i = 0; i < nS3; i++) latticepoint[index++] = 3;
           for(int i = 0; i < nS4; i++) latticepoint[index++] = 4;

           // Shuffle the latticepoint to randomize the spin states
           shuffle(latticepoint, numberspins);
       }
    
   
    
    
    //shuffle(latticepoint, numberspins);
    /*for (int i = 0; i < numberspins; i++)
      {   if(i==0) printf("{ {");
          printf("%4i ,", latticepoint[i]);
          if(i==numberspins-1) printf("}");
          if ((i + 1) % L1dim == 0)
              printf("}, \n");
      }*/
    //
	e = totalenergy();
    //printf("e = %4d",e);

	stdoutlog = fopen(stdoutlogname, "a");
	fprintf(stdoutlog, "Proc. %i: Initialized lattice with energy e=%i, create setup with %lf<e<%lf    %i    %i\n", myid, e, (emin + 0*(emax - emin) / 3), (emin + 3 * (emax - emin) / 3),Eminindex,Emaxindex);

	// run to a valid energy for the energy window
    
	// as long as energy is outside the middle third of local energy range
    
	int increm = 0;
//while ((e < (emin + 0*(emax - emin) / 3)) || (e > (emin + 3*(emax - emin) / 3)))------------------------------------------
    while ((e < emin) || (e > emax))
    {
        latticepoint[increm] = (q - 1);
        e = totalenergy();
        increm++;
    }
//---------------------------------------------------------------------------------------------------------------------------
	if (bctype == 1)
	{
		latticepoint[numberspins] = 1;         // Braskamp Kunz has fixed spin values at boundaries
		latticepoint[numberspins + 1] = -1;
	}

	// use last spin to store replica-id; this is just a marker to
	// allows us to keep track of replicas as they are exchanged
	latticepoint[numberspins + 2] = myid;

	// Pseudo WL Process is started for all processes to explore the energy levels
	stdoutlog = fopen(stdoutlogname, "a");
	//fprintf(stdoutlog, "\nPseudo WL Process has started for process: %i\n", myid);
	//fclose(stdoutlog);
    for (int i = 0; i <= Eglobalwidth; i++) lngE[i] = 0; //init H(E)
    //pseudowl();

	//Print lattice to logfile
	stdoutlog = fopen(stdoutlogname, "a");
	for (int i = 0; i < numberspins; i++)
    {   if(i==0) fprintf(stdoutlog, "{ {");
        fprintf(stdoutlog, "%4i ,", latticepoint[i]);
        if(i==numberspins-1) fprintf(stdoutlog, "}");
		if ((i + 1) % L1dim == 0)
			fprintf(stdoutlog, "}, \n {");
	}
	fprintf(stdoutlog, "\n");
	fprintf(stdoutlog, "%i", totalenergy());
	fprintf(stdoutlog, "\n");
	fclose(stdoutlog);
}

void init_hists() // initialize histograms
{
    lngE = (double*)malloc(hist_size * sizeof(double));
    lngE_buf = (double*)malloc(hist_size * sizeof(double));
    real_lngE = (double*)malloc(hist_size * sizeof(double));
    real_lngE_buf = (double*)malloc(hist_size * sizeof(double));
    microT = (double*)malloc(hist_size * sizeof(double));
    microT_buf = (double*)malloc(hist_size * sizeof(double));
    pseudolngE = (double*)malloc(hist_size * sizeof(double)); // added for pseudo wl process
    HE = (double*)malloc(hist_size * sizeof(double));

    for (int i = 0; i < hist_size; i++)
    {
        lngE[i] = 0.0;
        lngE_buf[i] = 0.0;
        real_lngE[i]= 0.0;
        real_lngE_buf[i] = 0.0;
        microT[i]= 0.0;
        microT_buf[i] = 0.0;
    }
}

void partial_init_hists() // initialize histograms
{
    for (int i = 0; i < hist_size; i++)
    {
        lngE_buf[i] = 0.0;
        real_lngE[i]= 0.0;
        real_lngE_buf[i] = 0.0;
        microT[i]= 0.0;
        microT_buf[i] = 0.0;
    }
}


// set boundaries for energy windows:
// either read from file (if precomputed for balanced REWL)
// or just split energy range evenly according to avaliable waklers
int find_local_energy_range(double Eglobmin, double Eglobmax, double overlap, int N)
{
	stdoutlog = fopen(stdoutlogname, "a");

	// consistency check
	if (N % multiple != 0)
	{
		fprintf(stdoutlog, "Total number of processes (%i) must be a multiple of the number of walkers per energy range (%i)!\n", N, multiple);
		fclose(stdoutlog);
		return (1);
	}

	FILE* window_init;

	// check if there is a file containing local energy window boundaries
	// Must be named Ewindows.dat !

    //fprintf(stdoutlog, "\nProc %i: Can't find file Ewindows.dat. Will calculate equal-size windows with overlap %lf\n", myid, overlap);
    //double Ewidth = (Eglobmax - Eglobmin) / (1.0 + ((double)(N / multiple) - 1.0) * (1.0 - overlap));
    //Emin = Eglobmin + (double)(myid / multiple) * (1.0 - overlap) * Ewidth;
    //Emax = Emin + Ewidth;
    //Eminindex = floor(Emin + -Eglobalmin);
    //Emaxindex = ceil(Emax + -Eglobalmin);

	if (fopen("Ewindows.dat", "r") == NULL)
	{
		fprintf(stdoutlog, "\nProc %i: Can't find file Ewindows.dat. Will calculate equal-size windows with overlap %lf\n", myid, overlap);
		double Ewidth = (Eglobmax - Eglobmin) / (1.0 + ((double)(N / multiple) - 1.0) * (1.0 - overlap));

		Emin = Eglobmin + (double)(myid / multiple) * (1.0 - overlap) * Ewidth;
		Emax = Emin + Ewidth;

		Eminindex = floor(Emin + -Eglobalmin);
		Emaxindex = ceil(Emax + -Eglobalmin);

		time(&timenow);
		fprintf(stdoutlog, "Proc %3i: Parameter: Eglobmin -- Eglobmax: %lf -- %lf; overlap=%i percent, Ewindowwidth=%lf, %s", myid, Eglobmin, Eglobmax, (int)(overlap * 100.0), Ewidth, ctime(&timenow));
        fprintf(stdoutlog, "Proc %3i: Parameter: Eminindex -- Emaxindex: %d -- %d;", myid, Eminindex, Emaxindex);
		fprintf(stdoutlog, "Proc %3i: Parameter: Emin -- Emax: %lf -- %lf; %s", myid, Emin, Emax, ctime(&timenow));
	}
	else // rewrite of prior file reading system to be more portable
	{
		FILE* fptr;
		int value;
		int valuee;
		int valueee;

		fptr = fopen("Ewindows.dat", "r");

		while (fscanf(fptr, "%i,%i,%i\n", &value, &valuee, &valueee) > 0)
		{
			if (value == myid / multiple)
			{
				Eminindex = valuee;
				Emaxindex = valueee;
				fprintf(stdoutlog, "%i %i %i\n", value, Eminindex, Emaxindex);
			}
			Emin = Eglobalmin + Eminindex;
			Emax = Eglobalmin + Emaxindex;
		}

		fclose(fptr);
	}

	fclose(stdoutlog);



	return (0);
}


// THIS IS THE MASTER RE / SWAP FUNCTION
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int replica_exchange(int swap_direction, int index_akt)
{
	int i_new; // histogram index of my configuration
	int Ecur; // current energy

	// frac refers to local exchange probability
	// wk is combined exchange probability
	double myfrac, otherfrac, randx, wk;

	int change = 0; // boolean: 0 for not exchanging, 1 for exchanging
	int swap_partner = -1; // id of swap partner (receive_buffer)

	swap_direction = swap_direction % 2; // comes actually as number of swap attempt

	// everyone has to find its swap-partner

	int* pairs; // array containing the partners of each process (send_buffer)
	pairs = (int*)malloc(2 * multiple * sizeof(int));

	if (mylocalid[swap_direction] == 0) // 'head-node' in the energy window determines pairs of flippartners
	{
		int choose_from = multiple; // number of free partners in higher window of communicator
		int select; // storage for random number

		int* libre; // list of free partners from higher window in communicator
		libre = (int*)malloc(multiple * sizeof(int));

		for (int i = 0; i < multiple; i++) libre[i] = multiple + i; // initialise

		// idea: processes from the lower window choose someone from the higher window at random
		// of course, the chosen walker can't have an exchange partner yet
		for (int i = 0; i < multiple; i++) // loop over processes in the lower window
		{
			select = rand() % choose_from;
			pairs[i] = libre[select];
			pairs[libre[select]] = i; // the 'vice-versa pair'
			// update list
			choose_from--;
			for (int j = select; j < choose_from; j++)
				libre[j] = libre[j + 1];
		}

		//       stdoutlog=fopen(stdoutlogname,"a");
		//       fprintf(stdoutlog,"Proc %3i: Drew the following swap partners:\n",myid);
		//       for (int i=0;i<2*multiple;i++)
		// 	fprintf(stdoutlog,"Proc %3i: %i -- %i (local ids in communicator)\n",myid,i,pairs[i]);
		//       fclose(stdoutlog);

		free(libre);
	}

	// at this point, every walker has a swap partner assigned, now they must be communicated
	if ((swap_direction == 0) && (myid < (numprocs - multiple))) // the walkers from the last node should not swap
	{
		comm_id = 2 * (myid / (2 * multiple)); // ! all integer, the '/' is a (div) ! Not the same as myid/multiple !
		MPI_Scatter(pairs, 1, MPI_INT, &swap_partner, 1, MPI_INT, 0, mpi_local_comm[comm_id]);
	}

	if ((swap_direction == 1) && (myid >= multiple)) // the walkers from the zero-node should not swap
	{
		comm_id = ((myid - multiple) / (2 * multiple)) * 2 + 1; // ! all integer, the '/' is a (div) ! See above
		MPI_Scatter(pairs, 1, MPI_INT, &swap_partner, 1, MPI_INT, 0, mpi_local_comm[comm_id]);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	free(pairs);

	if (swap_partner != -1) // i.e. if there is a swap-partner for me (if I am at a boundary, I might not have a swap partner this time)
	{
		// statistics
		if (swap_partner > mylocalid[swap_direction]) tryright++;
		else tryleft++;

		// safety cross check
		Ecur = totalenergy();
		if (Ecur + (-Eglobalmin) != index_akt)
		{
			stdoutlog = fopen(stdoutlogname, "a");
			fprintf(stdoutlog, "Proc %3i, replica_exchange(): Something is wrong here! Received index=%i, calculated index=%i totalenergy=%i. Abort.\n", myid, index_akt, Ecur + (-Eglobalmin), totalenergy());
			fclose(stdoutlog);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}


		i_new = index_akt;

		// get histogram index from my swap partner
		MPI_Sendrecv_replace(&i_new, 1, MPI_INT, swap_partner, 1, swap_partner, 1, mpi_local_comm[comm_id], &status);

		if ((i_new > Emaxindex) || (i_new < Eminindex)) // energyranges must overlap!
		{
			myfrac = -1.0;
		}
		else
		{
			// calculate my part of the exchange probability
			myfrac = exp(lngE[index_akt] - lngE[i_new]); // g(myE)/g(otherE)
		}

		if (mylocalid[swap_direction] < multiple) // I am receiver and calculator
		{
			// get my partners part of the exchange probability
			MPI_Recv(&otherfrac, 1, MPI_DOUBLE, swap_partner, 2, mpi_local_comm[comm_id], &status);

			// calculate combined exchange probability and roll the dice
			if ((myfrac > 0.0) && (otherfrac > 0.0))
			{
				randx = (1.0 * rand() / (RAND_MAX + 1.0));
				wk = myfrac * otherfrac;
				if (randx < wk) change = 1;
			}

			// tell my swap partner whether to exchange or not
			MPI_Send(&change, 1, MPI_INT, swap_partner, 3, mpi_local_comm[comm_id]);
		}
		else // I just send my part of exchange probability and await decision
		{
			MPI_Send(&myfrac, 1, MPI_DOUBLE, swap_partner, 2, mpi_local_comm[comm_id]);
			MPI_Recv(&change, 1, MPI_INT, swap_partner, 3, mpi_local_comm[comm_id], &status);
		}

		// if decision was made to exchange configurations
		if (change == 1)
		{
			// exchange spin conformations (incl. the 3 'special' lattice sites)
			MPI_Sendrecv_replace(&latticepoint[0], numberspins + 2 + 1, MPI_INT, swap_partner, 1, swap_partner, 1, mpi_local_comm[comm_id], &status);

			switched = i_new;

			// statistics
			if (swap_partner > mylocalid[swap_direction]) exchangeright++;
			else exchangeleft++;
		}
	}

	return(change);
	// returns whether or not configs were actually exchanged
}

void recombine(double countd)
{
    
    stdoutlog = fopen(stdoutlogname, "a");
    
    double init_dos = log(q)+100;
    
    int init_check=0;int init_check_2=0;
    double rec_lnge;double rec_real_lnge;
    int rec_en;int rec_en_other;
    int rec_en_2;int rec_en_other_2;
    double microT_compare;
       
    if (myid == 0) // 'root' in energy window, receive individual g(E) and send merged g(E)
    {
        for (int i = Eminindex; i <= Eglobalwidth; i++)
        {
            if (lngE[i] > 0.5 && init_check == 1)
            {
                real_lngE[i]=rec_real_lnge+(lngE[i]-rec_lnge);
                microT[i]=rec_real_lnge+((rec_lnge-lngE[i])/(rec_en-i));
                           
                rec_real_lnge=real_lngE[i];
                rec_lnge=lngE[i];
                rec_en = i;
            }
            if (lngE[i] > 0.5 && init_check == 0)
            {
                real_lngE[i]=init_dos;
                init_check=1;
                rec_real_lnge=real_lngE[i];
                rec_lnge=lngE[i];
                rec_en = i;
            }

        }
        
        //test output 1 for 0
        sprintf(filename, "1_Results/TestL%iq%i.HE.proc%04i.iter0", L1dim, q, myid);
        if ((file = fopen(filename, "w")) == NULL)
        {
            stdoutlog = fopen(stdoutlogname, "a");
            fprintf(stdoutlog, "Can not open file %s. Exit.\n", filename);
            fclose(stdoutlog);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        else
        {
            for (int i = Eminindex; i <= Eglobalwidth; i++)
            {
                if (lngE[i] > 0.5) fprintf(file, "%i\t%i\t%e\t%e\t%e\n", i, (i + (Eglobalmin)), lngE[i], real_lngE[i], microT[i]);
            }
            fclose(file);
        }
               
    
                   
        for (int ii = 1; ii < numprocs; ii++)
        {
            MPI_Recv(&lngE_buf[0], hist_size, MPI_DOUBLE, ii, 76, MPI_COMM_WORLD, &status); // get other dens. of states
            fprintf(stdoutlog, "\nProc %i: Received lngE from Proc. %i\n", myid, ii);
                       
            for (int i = Eminindex; i <= Eglobalwidth; i++)
            {
                if (real_lngE[i] > 0.5 && lngE_buf[i] > 0.5 && init_check_2 == 2)
                {
                    microT_buf[i]=real_lngE[rec_en_2]+((rec_lnge-lngE_buf[i])/(rec_en_2-i));
                               
                    if(abs(microT[i]-microT_buf[i])<microT_compare)
                    {
                        microT_compare=abs(microT[i]-microT_buf[i]);
                        rec_en_other_2=i;
                    }
                    rec_lnge=lngE_buf[i];
                    rec_en_2= i;
                               
                    /*fprintf(stdoutlog,"\n%i\n",i);
                    fprintf(stdoutlog,"\n%i\n",rec_en_other_2);
                    fprintf(stdoutlog,"\n%e\n",lngE_buf[i]);
                    fprintf(stdoutlog,"\n%f\n",microT[i]);
                    fprintf(stdoutlog,"\n%f\n",microT_buf[i]);
                    fprintf(stdoutlog,"\n%f\n", microT_compare);
                    */
                }
                if (real_lngE[i] > 0.5 && lngE_buf[i] > 0.5 && init_check_2 == 1)
                {
                    init_check_2 = 2;
                               
                    microT_buf[i]= real_lngE[rec_en_2]+((rec_lnge-lngE_buf[i])/(rec_en_2-i));
                    microT_compare=abs(microT[i]-microT_buf[i]);
                               
                    rec_lnge=lngE_buf[i];
                               
                    /*
                    fprintf(stdoutlog,"\n%i\n",i);
                    fprintf(stdoutlog,"\n%i\n",rec_en_2);
                    fprintf(stdoutlog,"\n%e\n",lngE_buf[i]);
                    fprintf(stdoutlog,"\n%f\n",microT[i]);
                    fprintf(stdoutlog,"\n%f\n",microT_buf[i]);
                    fprintf(stdoutlog,"\n%f\n", microT_compare);
                     */
                    rec_en_other_2=i;
                    rec_en_2= i;
                }
                if (real_lngE[i] > 0.5 && lngE_buf[i] > 0.5 && init_check_2 == 0)
                {
                    fprintf(stdoutlog,"\n%i\n",i);
                    rec_en_2 = i;
                    init_check_2 = 1;
                    rec_lnge=lngE_buf[i];
                }
                           
               }
                       
            init_check_2 = 0;
                       
            for (int i = rec_en_other_2; i <= Eglobalwidth; i++)
            {
                if (lngE_buf[i] > 0.5 && init_check_2 == 1)
                {
                    real_lngE[i]=rec_real_lnge+(lngE_buf[i]-rec_lnge);
                    microT[i]=rec_real_lnge+((rec_lnge-lngE_buf[i])/(rec_en_2-i));
                               
                    /*
                    fprintf(stdoutlog,"\ng %i\n",i);
                    fprintf(stdoutlog,"\n%i\n",rec_en_2);
                    fprintf(stdoutlog,"\n%e\n",rec_real_lnge);
                    fprintf(stdoutlog,"\n%e\n",lngE_buf[i]);
                    fprintf(stdoutlog,"\n%e\n",microT[i]);
                    fprintf(stdoutlog,"\n%f\n",rec_real_lnge+((rec_lnge-lngE_buf[i])/(rec_en_2-i)));
                    */
                               
                    rec_real_lnge=real_lngE[i];
                    rec_lnge=lngE_buf[i];
                    rec_en_2 = i;
                }
                if (lngE_buf[i] > 0.5 && init_check_2 == 0)
                {
                               
                    rec_real_lnge=real_lngE[i];
                    microT[i]=microT_buf[i];
                    
                    rec_en_2 = i;
                    init_check_2 = 1;
                    rec_lnge=lngE_buf[i];
                }
                           
            }
            for (int i = 0; i < hist_size; i++)
            {
                microT_buf[i] = 0.0;
                lngE_buf[i]=0.0;
            }
        }
                   
        sprintf(filename, "1_Results/Test2L%iq%i.HE.proc%04i.iter0", L1dim, q, myid);
        if ((file = fopen(filename, "w")) == NULL)
        {
            stdoutlog = fopen(stdoutlogname, "a");
            fprintf(stdoutlog, "Can not open file %s. Exit.\n", filename);
            fclose(stdoutlog);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        else
        {
            for (int i = Eminindex; i <= Eglobalwidth; i++)
            {
                if (real_lngE[i] > 0.5) fprintf(file, "%i\t%i\t%e\t%f\t%e\n", i, (i + (Eglobalmin)), lngE[i], real_lngE[i], microT[i]);
            }
            fclose(file);
        }
        
        sprintf(filename, "1_Results/Recombined_Output_%e.txt",countd);
        if ((file = fopen(filename, "w")) == NULL)
        {
            stdoutlog = fopen(stdoutlogname, "a");
            fprintf(stdoutlog, "Can not open file %s. Exit.\n", filename);
            fclose(stdoutlog);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        else
        {
            //fprintf(file, "{");
            for (int i = Eminindex; i <= Eglobalwidth; i++)
            {
                if (real_lngE[i] > 0.5) fprintf(file, "%i\t %f \n",(i +(Eglobalmin))+(L1dim*L1dim)*2, real_lngE[i]);
                
            }
            //fprintf(file, "}");
            fclose(file);
        }
                   
    }
    else // send individual g(E) and receive merged g(E)
    {
        fprintf(stdoutlog,"ufig");
        MPI_Send(&lngE[0], hist_size, MPI_DOUBLE, 0, 76, MPI_COMM_WORLD);
        fprintf(stdoutlog, "Proc %i: Sent lngE to Proc. %i\n", myid, 0);
    }
    fclose(stdoutlog);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char* argv[])
{
    //*********************************************************************************
    lowest_energy_lattice = (int*)malloc(numberspins * sizeof(int));
    //*********************************************************************************
    /*time(&timenow);
     std::cout << ctime(&timenow) << std::endl;*/
    
    clock_t tStart = clock();
    
    // set up MPI and MPI_COMM_WORLD
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    // check command line arguments
    sprintf(stdoutlogname, "1_Results/Error%04i.log", myid);
    if (argc < 5)
    {
        if (myid == 0)
        {
            stdoutlog = fopen(stdoutlogname, "w");
            fprintf(stdoutlog, "Unexpected number of command line arguments!\n");
            fprintf(stdoutlog, "Expect 4 arguments, %d were provided.\n", argc - 1);
            fprintf(stdoutlog, "Syntax: ./WLpotts_mpi [arg1] [arg2] [arg3] [arg4] \n\n");
            fprintf(stdoutlog, "Please provide the following command line arguments:\n");
            fprintf(stdoutlog, "1. Overlap between consecutive windows. [double, 0 <= overlap <= 1]\n");
            fprintf(stdoutlog, "2. Number of walkers per energy subwindow. [integer]\n");
            fprintf(stdoutlog, "3. Number of Monte Carlo steps between replica exchange. [integer]\n");
            fprintf(stdoutlog, "4. Random number seed. [integer]\n\n");
            fclose(stdoutlog);
            
            printf("ERROR: Unexpected number of command line arguments!\n");
            printf("       Expect 4 arguments, %d were provided.\n", argc - 1);
            printf("Syntax: ./WLpotts_mpi [arg1] [arg2] [arg3] [arg4] \n\n");
            printf("Please provide the following command line arguments:\n");
            printf("1. Overlap between consecutive windows. [double, 0 <= overlap <= 1]\n");
            printf("2. Number of walkers per energy subwindow. [integer]\n");
            printf("3. Number of Monte Carlo steps between replica exchange. [integer]\n");
            printf("4. Random number seed. [integer]\n\n");
            
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // number of walkers per energy window
    multiple = atoi(argv[2]);
    
    // at the moment, the code works only for an _odd_ number of energy windows
    // (to make the RE in windows at the outside consistent)
    if ((numprocs / multiple) % 2 == 0)
    {
        if (myid == 0)
        {
            stdoutlog = fopen(stdoutlogname, "a");
            fprintf(stdoutlog, "ERROR: Even number of energy windows (%d) requested. Please request an odd number of energy windows.\n\n", numprocs / multiple);
            fclose(stdoutlog);
            
            printf("ERROR: Even number of energy windows (%d) requested. Please request an odd number of energy windows.\n\n", numprocs / multiple);
        }
        
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    //MPI_Barrier(MPI_COMM_WORLD);
    
    // initiate random numbers
    rseed = atoi(argv[4]);
    srand(rseed + myid);
    
    int swap_every = atoi(argv[3]);      // after this number of sweeps try conformations swap
    int swap_every_init = swap_every;
    
    // init log file
    sprintf(stdoutlogname, "1_Results/Proc%04i.sim%i.log", myid, rseed);
    
    // set local energy range
    ret_status = find_local_energy_range(Eglobalmin, Eglobalmax, atof(argv[1]), numprocs);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (ret_status != 0)
    {
        stdoutlog = fopen(stdoutlogname, "a");
        fprintf(stdoutlog, "Proc. %3i: find_local_energy_range() returned %i\n", myid, ret_status);
        fclose(stdoutlog);
        
        MPI_Abort(MPI_COMM_WORLD, 1);    // something went wrong in find_local_energy_range()
    }
    
    init_hists(); // moved above init_lattice() for calculation considerations
    init_neighbors();
    init_lattice(Emin, Emax); // 0 - random; 1 - all equal
    
    // calculate energy for the first time
    int eold, energie;
    energie = totalenergy();
    
    
    
    
    
    
    
    stdoutlog = fopen(stdoutlogname, "a");
    fprintf(stdoutlog, "Proc %3i: energy at start=%i\n", myid, energie);
    fclose(stdoutlog);
    energie -= Eglobalmin; // shift to positive values to use it as array index
    
    Estartindex = energie;
    
    int stop = 0;
    stdoutlog = fopen(stdoutlogname, "a");
    fprintf(stdoutlog, "Proc %3i: Parameter: Eminindex -- Emaxindex: %i -- %i; Estartindex=%i\t", myid, Eminindex, Emaxindex, Estartindex);
    if ((Estartindex > Emaxindex) || (Estartindex < Eminindex))
    {
        fprintf(stdoutlog, "Start out of range!!!\n");
        stop = 1;
    }
    else
    {
        fprintf(stdoutlog, "OK\n");
    }
    
    fclose(stdoutlog);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (stop) MPI_Abort(MPI_COMM_WORLD, 1);
    
    //Teststop
    //   MPI_Barrier(MPI_COMM_WORLD);
    //   MPI_Abort(MPI_COMM_WORLD,1);
    
    // create new groups and communicators for each energy range window
    stdoutlog = fopen(stdoutlogname, "a");
    MPI_Comm_group(MPI_COMM_WORLD, &world); // get the group of processes in MPI_COMM_WORLD (i.e. all)
    int* ranks;
    ranks = (int*)malloc(2 * multiple * sizeof(int));
    mpi_local_group = (MPI_Group*)malloc(((numprocs / multiple) - 1) * sizeof(MPI_Group));
    mpi_local_comm = (MPI_Comm*)malloc(((numprocs / multiple) - 1) * sizeof(MPI_Comm));
    
    for (int i = 0; i < ((numprocs / multiple) - 1); i++) // i is counter for the energy range windows
    {
        for (int j = 0; j < 2 * multiple; j++)
        {
            ranks[j] = i * multiple + j; // contains the ranks of processes in MPI_COMM_WORLD which should get into local group
            if (myid == 0)
            {
                fprintf(stdoutlog, "Proc %3i: %i will be part of communicator/group %i\n", myid, ranks[j], i);
            }
        }
        MPI_Group_incl(world, 2 * multiple, ranks, &mpi_local_group[i]); // create local group
        MPI_Comm_create(MPI_COMM_WORLD, mpi_local_group[i], &mpi_local_comm[i]); // create communicator for that group
    }
    
    free(ranks);
    
    // get my local id (in my local communicators)
    if (myid < numprocs - multiple)
    {
        comm_id = 2 * (myid / (2 * multiple));
        MPI_Comm_rank(mpi_local_comm[comm_id], &mylocalid[0]);
        fprintf(stdoutlog, "Proc %3i: I am part of communicator/group %i with local_id[0]=%i\n", myid, comm_id, mylocalid[0]);
    }
    else
    {
        mylocalid[0] = INT_MAX; // just to give it a value
        fprintf(stdoutlog, "Proc %3i: got local_id[0]=%i\n", myid, mylocalid[0]);
    }
    
    if (myid >= multiple)
    {
        comm_id = 2 * ((myid - multiple) / (2 * multiple)) + 1;
        MPI_Comm_rank(mpi_local_comm[comm_id], &mylocalid[1]);
        fprintf(stdoutlog, "Proc %3i: I am part of communicator/group %i with local_id[1]=%i\n", myid, comm_id, mylocalid[1]);
    }
    else
    {
        mylocalid[1] = INT_MAX; // just to give it a value
        fprintf(stdoutlog, "Proc %3i: got local_id[1]=%i\n", myid, mylocalid[1]);
    }
    
    
    // log 'path'data (the same data is written, just differently sorted: with respect to the sample id and with respect to process id)
    sprintf(wanderlogname, "1_Results/wanderlog_rseed%i_sample_%i.dat", rseed, latticepoint[numberspins + 2]);
    wanderlog = fopen(wanderlogname, "w");
    time(&timenow);
    fprintf(wanderlog, "#sweep\t\tproc-id\tconf-id\tenergy\t\ttime\n");
    fprintf(wanderlog, "%e\t%i\t%i\t%i\t%s", 0.0, myid, latticepoint[numberspins + 2], energie, ctime(&timenow));
    fclose(wanderlog);
    
    sprintf(wanderlogname, "1_Results/wanderlog_rseed%i_walker_%i.dat", rseed, myid);
    wanderlog = fopen(wanderlogname, "w");
    time(&timenow);
    fprintf(wanderlog, "#sweep\t\tproc-id\tconf-id\tenergy\t\ttime\n");
    fprintf(wanderlog, "%e\t%i\t%i\t%i\t%s", 0.0, myid, latticepoint[numberspins + 2], energie, ctime(&timenow));
    fclose(wanderlog);
    
    //start simulation
    double wsk, dice; // wsk: probability
    int wiggle;
    int wiggletwo;   // ID of spin to be updated (only single-spin update implemented)
    
    // the WL parameter should eventually go into the init file, too
    double countdown = 10;
    double lnf =1;                    // my modification factor////////////////////////////////////////////////////////////////////////////
    double lnf_slowest = lnf;            // modification factor of slowest walker in my window
    //double lnfmin=log(1.000000001);
    double lnfmin = log(1.00001);      // terminal modification factor
    double sweep = 0;                    // counter for MC sweeps
    int flat;                          // 0 - histogram not flat; 1 - flat
    int iteration = 1;                   // WL iteration counter
    int iter = 1;
    double check_flatness_every = 10000;   // in number of sweeps
    int backup;
    int backuptwo;
    double update_enegy, updateold;
    long int count_neigh=0;
    double ads_config[Eglobalwidth];
    double ads_count[Eglobalwidth];

	eold = energie;

	int swtch;
	int found = 0;
    int allowed_lattices = 30;
    int num_lattices;
    
   
    FILE *file_ptr = fopen("1_Results/0_Lattices.txt", "w");
	fprintf(stdoutlog, "Proc %3i: Start WL iteration\n", myid);
	fclose(stdoutlog);

	for (int i = 0; i <= Eglobalwidth; i++) 
    {
        HE[i] = 0; //init H(E)
        //ads_config[i] = 0;
    }
        

	// Main Wang-Landau routine
	while (lnf_slowest > lnfmin)
	{
        for (int k = 0; k < check_flatness_every; k++)
        {
            for (int i = 0; i < numberspins; i++) // this does 1 MC sweep
            {
                int wiggle = rand() % numberspins; // choose spin to wiggle at random
                int backup = latticepoint[wiggle]; // remember old spin orientation
                int swap_with_backup; // will use this to store the index of the spin we swap with, if the swap is rejected
                int update_energy = propose_update(wiggle, &swap_with_backup); // the function now also returns the swap_with index
                int energie = eold + update_energy; // calculate energy of updated configuration
                // reject because you got out of bounds of energy window
                if ((energie > Emaxindex) || (energie < Eminindex))
                {
                    int temp = latticepoint[wiggle];
                    latticepoint[wiggle] = latticepoint[swap_with_backup];
                    latticepoint[swap_with_backup] = temp;
                    energie = eold;
                }
                else
                { // calculate acceptance probability
                    double dice = (double)rand() / (RAND_MAX + 1.0); // roll the dice
                    double wsk = exp(lngE[eold] - lngE[energie]); // calculate acceptance probability
                    if (dice > wsk)
                    { // reject
                        // swap the spins back
                        int temp = latticepoint[wiggle];
                        latticepoint[wiggle] = latticepoint[swap_with_backup];
                        latticepoint[swap_with_backup] = temp;
                        energie = eold;
                        // energie is already set to eold
                    }
                    else
                    {
                        eold = energie; // accept
                    }
                }                        // update histograms
                lngE[energie] += lnf;
                HE[energie]++;
                //int result = check_neighbor_spins(latticepoint, neighbor, numberspins);
                int result = count_spin_1_neighbors_of_spin_0(latticepoint, neighbor, numberspins);
                ads_count[energie]+=1;
                if (result != 0)
                {
                    //printf("result = %d \n",result);
                    memcpy(lowest_energy_lattice, latticepoint, numberspins * sizeof(int));
                    
                    ads_config[energie] +=result;
                    //lngE[energie] += lnf;
                    //HE[energie]++;
                    //printf(" {%d, %f }\n",energie,log(ads_config[energie]));
                }
                
               
            }
            
            //==========================================================================================
            
            
            if (num_lattices <= allowed_lattices - 1) {
                if (file_ptr == NULL) {
                    printf("Error opening file!\n");
                    return 1; // Or handle the error as you see fit
                }

                if (num_lattices == 0) {
                    fprintf(file_ptr, "{");
                }

                fprintf(file_ptr, "{"); // Begin matrix

                for (int i = 0; i < numberspins; i++) {
                    if (i % L1dim == 0) {
                        if (i != 0) {
                            fprintf(file_ptr, "},\n"); // End of a row (except for the first row)
                        }
                        fprintf(file_ptr, "{"); // Begin a new row
                    } else {
                        fprintf(file_ptr, ", "); // Element separator within a row
                    }

                    fprintf(file_ptr, "%d", lowest_energy_lattice[i]); // Print the element
                }

                fprintf(file_ptr, "}} \n");
                num_lattices += 1;

                if (num_lattices < allowed_lattices - 1) {
                    fprintf(file_ptr, ",");
                }
            }

               
            //==========================================================================================
            //********************************************************************************************************
            
            /*if (eold < lowest_energy) {
                lowest_energy = eold;
                memcpy(lowest_energy_lattice, latticepoint, numberspins * sizeof(int));
            }*/
            //if (iteration == 10)
           
               
            
          
        
            //********************************************************************************************************
            /*int result = check_spin_counts(latticepoint, numberspins, nS0, nS1, nS2, nS3, nS4);
            if (result == 0) {
                printf("Spin counts are not correct. %4d \n",k);
            }*/
            /*if (result == 1) {
                printf("Spin counts are correct. %4d =================================\n",k);
            }*/
            /*for (int i = 0; i < numberspins; i++)
              {   if(i==0) printf("{ {");
                  printf("%4i ,", latticepoint[i]);
                  if(i==numberspins-1) printf("}");
                  if ((i + 1) % L1dim == 0)
                      printf("}, \n");
              }
              printf("\n \n");*/
            //==========================================================================================
			sweep++; // sweep counter

			swap_every--; // RE counter (countdown)
			if (swap_every == 0) // ignition! (time to try to exchange configurations)
			{
				swap_every = swap_every_init; // reset countdown clock
				if (replica_exchange(sweep / swap_every, eold) == 1) // if configs were exchanged ('accept'-branch)
				{
					energie = switched;
					eold = energie;
					stdoutlog = fopen(stdoutlogname, "a");
					fprintf(stdoutlog, "Proc %3i: %i iteration, %e sweeps, Replica Exchange Success\n Energy Calculted: %i \t Recievced Index Energy: %i\n", myid, iteration, sweep, energie, switched);
					fclose(stdoutlog);
				}
				// update histograms (independently of whether RE happened or not)
				lngE[energie] += lnf;
				HE[energie]++;
                //int result = check_neighbor_spins(latticepoint, neighbor, numberspins);
                int result = count_spin_1_neighbors_of_spin_0(latticepoint, neighbor, numberspins);
                if (result != 0)
                {
                    memcpy(lowest_energy_lattice, latticepoint, numberspins * sizeof(int));
                    ads_config[energie] +=result;
                    //lngE[energie] += lnf;
                    //HE[energie]++;
                    //printf(" {%d, %f }\n",energie,ads_config[energie]);
                }

				// record position (process) of the actual sample
				sprintf(wanderlogname, "1_Results/wanderlog_rseed%i_sample_%i.dat", rseed, latticepoint[numberspins + 2]);
				wanderlog = fopen(wanderlogname, "a");
				time(&timenow);
				fprintf(wanderlog, "%e\t%i\t%i\t%i\t%s", sweep, myid, latticepoint[numberspins + 2], eold, ctime(&timenow));
				fclose(wanderlog);

				// record actual sample id for each walker
				sprintf(wanderlogname, "1_Results/wanderlog_rseed%i_walker_%i.dat", rseed, myid);
				wanderlog = fopen(wanderlogname, "a");
				time(&timenow);
				fprintf(wanderlog, "%e\t%i\t%i\t%i\t%s", sweep, myid, latticepoint[numberspins + 2], eold, ctime(&timenow));
				fclose(wanderlog);
			}
		} // end computation block

        sprintf(filename, "1_Results/Prelim.L%iq%i.HE.proc%04i.iter%i", L1dim, q, myid, iteration); // Preliminary output file that allows for process inspection
		sprintf(filename, "1_Results/Prelim.L%iq%i.HE.proc%04i.iter%i", L1dim, q, myid, iteration); // Preliminary output file that allows for process inspection
		if ((file = fopen(filename, "w")) == NULL)
		{
			stdoutlog = fopen(stdoutlogname, "a");
			fprintf(stdoutlog, "Can not open file %s. Exit.\n", filename);
			fclose(stdoutlog);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		else
		{
			for (int i = Eminindex; i <= Emaxindex; i++)
			{
				fprintf(file, "%i\t%i\t%e\t%e\t%e\n", i, (i + (Eglobalmin)), 0.0, lngE[i], HE[i]);
			}
			fclose(file);
		}

		// 1_check flatness////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		flat = histflat(Eminindex, Emaxindex, 0.8);
		if (flat == 0) // histogram not flat, do nothing except maybe some logging
		{
			stdoutlog = fopen(stdoutlogname, "a"); // Prints out effictively flatness progress for inspection
			fprintf(stdoutlog, "Proc %3i: %i iteration, %e sweeps, Histogram noch nicht flach\nValue of Tried Minimum Histogram Value: %f    Value of Tried Ratio: %f\n", myid, iteration, sweep, flatmin, flatratio);

			phrepv = flatmin; // holds the minimum value of the histogram and checks change and prints out the current lattice if the value has reoccured frequently
			if (repv == phrepv)
			{
				iterr++;
				repv = phrepv;
			}
			else
			{
				repv = phrepv;
			}

			if (iterr == 20)
			{
				iterr = 0;

				fprintf(stdoutlog, "\n");

				/*for (int i = 0; i < numberspins; i++)
				{
					fprintf(stdoutlog, "%4i , ", latticepoint[i]);
					if ((i + 1) % L1dim == 0)
						fprintf(stdoutlog, "\n");

				}*/
                fprintf(stdoutlog, "{"); // Start of the matrix
                for (int i = 0; i < numberspins; i++)
                {
                    // Print the lattice point
                    fprintf(stdoutlog, "%4i", latticepoint[i]);

                    // Add a comma after each element except the last one in the matrix
                    if (i < numberspins - 1)
                        fprintf(stdoutlog, ", ");

                    // Line break and open a new row in the matrix after each L1dim elements
                    if ((i + 1) % L1dim == 0)
                    {
                        if (i < numberspins - 1)
                            fprintf(stdoutlog, "},\n{"); // Close the current row and start a new one
                        else
                            fprintf(stdoutlog, "}"); // Close the last row
                    }
                }
                fprintf(stdoutlog, "}\n"); // End of the matrix

				fprintf(stdoutlog, "\n System Energy Index: %i \n", energie);

				fprintf(stdoutlog, "\n");
			}

			fclose(stdoutlog);
		}
		else // histograms of all walkers are 'flat'
		{
			if (lnf > lnfmin) // as long as modification factor is still larger than terminal value
			{
				sprintf(filename, "1_Results/L%iq%i.HE.proc%04i.iter%i", L1dim, q, myid, iteration);
				if ((file = fopen(filename, "w")) == NULL)
				{
					stdoutlog = fopen(stdoutlogname, "a");
					fprintf(stdoutlog, "Can not open file %s. Exit.\n", filename);
					fclose(stdoutlog);
					MPI_Abort(MPI_COMM_WORLD, 1);
				}
				else
				{
					for (int i = Eminindex; i <= Emaxindex; i++)
					{
						if (HE[i] > 0.5) fprintf(file, "%i\t%i\t%e\t%e\t%e\n", i, (i + (Eglobalmin)), 0.0, lngE[i], HE[i]);
					}
					fclose(file);
				}
			}

			// decrease modification factor
			// canonical method (reduce by sqrt(2)) implemented
			lnf /= 2.0;

			for (int i = 0; i <= Eglobalwidth; i++) 
            {
                HE[i] = 0; //reset H(E)
                //ads_config[i] =0;
            }
			iteration++; // iteration counter

			if (merge_hists == 1) // merge g(E) estimators from multiple walkers in the same energy window
			{
				stdoutlog = fopen(stdoutlogname, "a");
				if (myid % multiple == 0) // 'root' in energy window, receive individual g(E) and send merged g(E)
				{
					for (int i = 1; i < multiple; i++)
					{
						MPI_Recv(&lngE_buf[0], hist_size, MPI_DOUBLE, myid + i, 77, MPI_COMM_WORLD, &status); // get other dens. of states
						fprintf(stdoutlog, "Proc %i: Received lngE from Proc. %i\n", myid, myid + i);
						for (int j = 0; j < hist_size; j++) lngE[j] += lngE_buf[j]; // sum up for average
					}
					for (int j = 0; j < hist_size; j++) lngE[j] /= (double)multiple; // normalize
					for (int i = 1; i < multiple; i++)
					{
						MPI_Send(&lngE[0], hist_size, MPI_DOUBLE, myid + i, 99, MPI_COMM_WORLD);
						fprintf(stdoutlog, "Proc %i: Sent merged lngE to Proc. %i\n", myid, myid + i);
					}
				}
				else // send individual g(E) and receive merged g(E)
				{
					MPI_Send(&lngE[0], hist_size, MPI_DOUBLE, myid - (myid % multiple), 77, MPI_COMM_WORLD);
					fprintf(stdoutlog, "Proc %i: Sent lngE to Proc. %i\n", myid, myid - (myid % multiple));
					MPI_Recv(&lngE_buf[0], hist_size, MPI_DOUBLE, myid - (myid % multiple), 99, MPI_COMM_WORLD, &status);
					fprintf(stdoutlog, "Proc %i: Received merged lngE from Proc. %i\n", myid, myid - (myid % multiple));
					for (int j = 0; j < hist_size; j++) lngE[j] = lngE_buf[j]; // replace individual lngE (could be done directly, yes)
				}
				fclose(stdoutlog);
			}
		}

		// check progress from all other windows
		MPI_Allreduce(&lnf, &lnf_slowest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if(lnf_slowest<=(((double) (10000000/countdown))*.0000001))
        {
            recombine((((double) (10000000/countdown))*.0000001));
            countdown *= 10;
            partial_init_hists() ;
            /*if(lnf_slowest<=lnfmin)
            {
                printf("%i lnf = %e \n",iteration,lnf);
            }*/
        }
        
		// just some logging
		if (flat == 1)
		{
			stdoutlog = fopen(stdoutlogname, "a");
			fprintf(stdoutlog, "Proc %3i: Start %i iteration, %e sweeps total so far, lnf now %e (lnfmin=%.2e, lnf_slowest=%e)\n", myid, iteration, sweep, lnf, lnfmin, lnf_slowest);
			fprintf(stdoutlog, "Proc %3i: tryleft: %i, exchangeleft %i (Akzeptanzleft:%.2lf) <--> tryright: %i, exchangeright %i (Akzeptanzright:%.2lf)\n", myid, tryleft, exchangeleft, (double)exchangeleft / (double)tryleft, tryright, exchangeright, (double)exchangeright / (double)tryright);
			fclose(stdoutlog);
		}
        //printf("energie = %i \n",eold);
        
        
	}// end while(lnf_slowest>lnfmin) -> this terminates the simulation
    fprintf(file_ptr, "}");
    fclose(file_ptr);
    //printf(" Energy: %d\n", lowest_energy);
    /*printf("Lattice Configuration:\n");
    printf("{"); // Begin matrix
    for (int i = 0; i < numberspins; i++) {
        if (i % L1dim == 0) {
            if (i != 0) {
                printf("},\n"); // End of a row (except for the first row)
            }
            printf("{"); // Begin a new row
        } else {
            printf(", "); // Element separator within a row
        }
        printf("%d", lowest_energy_lattice[i]); // Print the element
    }
    printf("}} \n \n"); // Close the last row and the matrix*/
     
    // ************************************************************************************************
    sprintf(adslat, "1_Results/1_Ads_DoS_lattice_%i.txt", L1dim);
    adslatlog = fopen(adslat, "w");
    if (adslatlog != NULL) {
        for (int i = Eminindex; i <= Emaxindex; i++) {
            // Check if the element is within the desired range and is an integer
            if (ads_config[i] >= 1.0 && ads_config[i] <= exp(20) && floor(ads_config[i]) == ads_config[i] ) {
                fprintf(adslatlog, "%i\t%f\n",(i +(Eglobalmin))+(L1dim*L1dim)*2, ads_config[i]/ads_count[i]);
            }
        }
        fclose(adslatlog);
    } else {
        printf("Error opening file %s for writing.\n", adslat);
    }
    // ************************************************************************************************



    



//   free(neighbor);
//   free(latticepoint);

  // normalize results
	double norm = lngE[Eminindex] - log(q);
	for (int i = 0; i <= 2 * Eglobalwidth; i++) lngE[i] -= norm;

	// write final data to file
	sprintf(filename, "1_Results/L%iq%i.proc%04i.lngE", L1dim, q, myid);
	if ((file = fopen(filename, "w")) != NULL)
	{
		for (int i = Eminindex; i <= Emaxindex; i++)
			fprintf(file, "%i\t%i\t%e\t%e\t%e\n", i, (i + Eglobalmin), 0.0, lngE[i]-lngE[0]+log(q), HE[i]);
		fclose(file);
	}
    /******************************************************************************************************************************************************/
    free(lowest_energy_lattice);
    /******************************************************************************************************************************************************/
	//   free(HE);
	//   free(lngE);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	/*time(&timenow);
	std::cout << ctime(&timenow) << std::endl;*/

	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	return(0); // you did it!
}
