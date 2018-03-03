#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "iostream"
#include "common.h"
#include "omp.h"

//
//  benchmarking program
//
using namespace std;
int main( int argc, char **argv ) {
    int navg, nabsavg = 0;
    double davg, dmin, absmin = 1.0, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 8000);

    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;
    int binslength = 0;
    //
    set_size(n, binslength);
    int nn = sqrt(binslength);

    vector<vector<particle_t *> > bins(binslength);

    for (int i = 0; i < binslength; ++i) {
        bins.push_back(vector<particle_t *>());
    }
    particle_t *particles = (particle_t *) malloc(n * sizeof(particle_t));

    init_particles(n, particles, bins);

    //
    //  simulate a number of time steps
    //
    int numthreads;
    double simulation_time = read_timer();
    #pragma omp parallel
    {
        numthreads = omp_get_num_threads();
    }

    {
    for (int step = 0; step < NSTEPS; step++) {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;
        //
        //  compute forces
        //
        #pragma omp  parallel for //schedule(auto) reduction (+:davg) reduction(+:navg)
        for (int l = nn + 1; l < binslength - nn; l += nn) {


            for (int i = 0 ;i < nn - 2; ++i) {
                int pos1 = i+l;
                unsigned long currentbinlength = bins[pos1].size();


                int pos;
                particle_t *p;
                for (unsigned long j = 0; j < currentbinlength; ++j) {


                    pos = pos1;
                    //std::advance(p, j);
                     p = bins[pos1][j];
                    (*p).ay = (*p).ax = 0;

                    //apply forces to particle from same bin


                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //apply forces to particle from surrounding bins

                    //bin to the left
                    pos = pos1 - 1;


                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin to the right
                    pos = pos1 + 1;
                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin above and left
                    pos = pos1 - nn - 1;

                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin above
                    pos = i + l - nn;

                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin above and right
                    pos = pos1 - nn + 1;
                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin below and left
                    pos = pos1 + nn - 1;
                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin below
                    pos = pos1 + nn;
                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                    //bin below and right
                    pos = pos1 + nn + 1;
                    for (int k =0, length = bins[pos].size(); k < length; ++k) {
                        apply_force(*p, *bins[pos][k], &dmin, &davg, &navg);

                    }

                }

            }
        }

        #pragma omp  for schedule(dynamic)
        for (int i = 0; i < binslength; ++i) {
            vector<particle_t*> temp;
            //temp.reserve(20);
            bins[i].swap(temp);
            //bins[i].clear();
        }


        //bins = vector<list<particle_t *> >(binslength);;

        #pragma omp  for simd
        for (int i = 0; i < n; ++i) {
            move(particles[i], bins);
        }
        //
        //  move particles
        //
        if (find_option(argc, argv, "-no") == -1) {
            //
            // Computing statistical data
            //

            if (navg) {
                absavg += davg / navg;
                nabsavg++;
            }

            if (dmin < absmin) absmin = dmin;

            //
            //  save if necessary
            //

            if (fsave && (step % SAVEFREQ) == 0)
                save(fsave, n, particles);
        }
    }
}
    simulation_time = read_timer( ) - simulation_time;

    printf("n = %d, threads = %d, simulation time = %g seconds",n,numthreads,simulation_time);
    if( find_option( argc, argv, "-no" ) == -1 )
    {
        if (nabsavg) absavg /= nabsavg;
        //
        //  -the minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );
    free( particles );
    if( fsave )
        fclose( fsave );

    return 0;
}
