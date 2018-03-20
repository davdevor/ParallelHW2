#include <mpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "commonmpi.h"


//
//  benchmarking program
//
using namespace std;
int main( int argc, char **argv )
{
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    int binslength = 0;
    set_size( n , binslength);
    int nn = sqrt(binslength);




    vector<list<particle_t*> > bins;
    particle_t *particles = (particle_t *) malloc(n * sizeof(particle_t));

    int *localCount = (int*) malloc(sizeof(int));
    particle_t *tempP = (particle_t *) malloc(n * sizeof(particle_t));
    particle_t *local = (particle_t *) malloc(n * sizeof(particle_t));
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    int *partition_offsets = (int*) malloc( n_proc+1 * sizeof(int) );
    bins.reserve(binslength);
    for (int i = 0; i < binslength; ++i) {
        bins.push_back(list<particle_t *>(0));
    }
    partition_offsets[1] = 0;
    partition_sizes[0] = 0;
    //get particles out of bins and order them in a list to be sent to other processes
    if(rank == 0) {
        init_particles(n, particles, bins);
        int lengthper = nn/n_proc;
        for(int i = 1; i < n_proc; ++i){
            int count = 0;
            for(int j = 0; j < lengthper; ++j){
                if(j == lengthper-nn){
                  partition_offsets[i+1] = count;
                }
                //x*rowSize + y
                for(int index = 0; index < nn; ++index){
                while(bins[(i+j)*nn+index].size()>0) {
                    tempP[j + partition_sizes[i - 1]] = *bins[(i+j) * nn + index].front();
                    bins[(i+j) * nn + index].pop_front();
                    ++count;
                }
                }
            }
            //cout<<"COUNT " << count<<" RANK" << rank <<endl;
            partition_sizes[i] = count;
        }
        //send to each process how many particles to receive
        int *x = new int;
        int y = 1;
        for(int i =1; i < n_proc; ++i){
            *x = partition_sizes[i]+(partition_sizes[i-1]-partition_offsets[i]);
            MPI_Send(x,y,MPI_INT,i,0,MPI_COMM_WORLD);
            //cout<<*x<<" send "<<i<<endl;
        }
        //cout << "after first send"<<endl;
        int index = 0;
        for(int i = 0; i < binslength; ++i){
            while(bins[i].size()>0){
                local[index] = *bins[i].front();
                bins[i].pop_front();
                index++;
            }

        }
        //cout<<"end if rank 0"<<endl;
    }
    else{
        int y = 1;
        MPI_Recv(localCount,y,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //cout <<"after first recv "<<*localCount<<endl;

    }
    if(rank == 0){
        particle_t  *temp = tempP;
        //send particles
        int y;
        for(int i = 1; i < n_proc; ++i){
            //cout<<partition_offsets[i]<<endl;
            temp+=partition_offsets[i];
            y = partition_sizes[i]+(partition_sizes[i-1]-partition_offsets[i]);
            MPI_Send(temp,y,PARTICLE,i,0,MPI_COMM_WORLD);
        }
        //cout<<"after send particles"<<endl;
    }
    else{
        //cout<<"Local "<< *localCount <<endl;
        MPI_Recv(local,*localCount,PARTICLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        //cout<<"after 2nd rec"<<endl;
        //put particles in local bins
        particle_t p;
        for(int i =0; i < *localCount; ++i){
            p= local[i];

            int x = floor(p.x/myconst)+1;
            int y = floor(p.y/myconst)+1;
            bins[x*nn + y].push_front(&p);
        }
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    {
        for (int step = 0; step < 100; step++) {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            if( find_option( argc, argv, "-no" ) == -1 )
                if( fsave && (step%SAVEFREQ) == 0 )
                    save( fsave, n, particles );

            //
            //  compute forces
            //
            int currentbinlength;
            int pos;
            std::list<particle_t *>::iterator it;
            std::list<particle_t *>::iterator end;
            for (int l = nn + 1; l < binslength - nn; l += nn) {
                for (int i = 0, pos1 = l; i < nn - 2; ++i, ++pos1) {
                    currentbinlength = bins[pos1].size();
                    //loop trough all particles in current bin
                    for (int j = 0; j < currentbinlength; j++) {
                        pos = pos1;
                        particle_t *p = bins[pos].back();
                        bins[pos].pop_back();
                        p->ay = p->ax = 0;

                        //apply forces to particle from same bin

                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //apply forces to particle from surrounding bins

                        //bin to the left
                        pos = pos1 - 1;

                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin to the right
                        pos = pos1 + 1;
                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin above and left
                        pos = pos1 - nn - 1;
                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin above
                        pos = i + l - nn;
                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin above and right
                        pos = pos1 - nn + 1;
                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin below and left
                        pos = pos1 + nn - 1;
                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin below
                        pos = pos1 + nn;
                        it = bins[pos].begin();
                        for (; it != bins[pos].end(); ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }

                        //bin below and right
                        pos = pos1 + nn + 1;
                        it = bins[pos].begin();
                        end = bins[pos].end();
                        for (; it != end; ++it) {
                            apply_force(*p, **it, &dmin, &davg, &navg);
                        }
                        bins[pos1].push_front(p);
                    }

                }
            }
            for (int i = 0; i < binslength; ++i) {
                bins[i].clear();
            }



            //
            //  move particles
            //
            for (int i = 0; i < *localCount; i++)
                move(local[i]);


            //DIVIDE PARTICLES BACK UP


            //send all particles back to root
            MPI_Request sendRequest;
            if(rank != 0){
                //cout<<"main send"<<endl;
                MPI_Isend(local,*localCount,PARTICLE,0,0,MPI_COMM_WORLD,&sendRequest);
                //MPI_Isend(local,*localCount,PARTICLE,0,0,MPI_COMM_WORLD);
            }
            else if(rank ==0){
                //cout<<"rank main loop "<<rank<<endl;
                particle_t  *temp = tempP;
                //send particles
                int y;
                for(int i = 1; i < n_proc; ++i){
                    temp+=partition_offsets[i];
                    y = partition_sizes[i]+(partition_sizes[i-1]-partition_offsets[i]);
                    MPI_Recv(temp,y,PARTICLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
                //cout<<"finished recv"<<endl;

            }
            //MPI_Wait(&sendRequest,MPI_STATUS_IGNORE);
            //put particles into bins and thenget particles out of bins and order them in a list to be sent to other processes
            if(rank == 0) {
                particle_t p;
                for(int i =0; i < n; ++i){
                    p= tempP[i];

                    int x = floor(p.x/myconst)+1;
                    int y = floor(p.y/myconst)+1;
                    bins[x*nn + y].push_front(&p);
                }
                int lengthper = nn/n_proc;
                for(int i = 1; i < n_proc; ++i){
                    int count = 0;
                    for(int j = 0; j < lengthper; ++j){
                        if(j == lengthper-nn){
                            partition_offsets[i+1] = count;
                        }
                        //x*rowSize + y
                        for(int index = 0; index < nn; ++index){
                            while(bins[(i+j)*nn+index].size()>0) {
                                tempP[j + partition_sizes[i - 1]] = *bins[(i+j) * nn + index].front();
                                bins[(i+j) * nn + index].pop_front();
                                ++count;
                            }
                        }
                    }
                    //cout<<"COUNT " << count<<" RANK" << rank <<endl;
                    partition_sizes[i] = count;
                }
                //send to each process how many particles to receive
                int *x = new int;
                int y = 1;
                for(int i =1; i < n_proc; ++i){
                    *x = partition_sizes[i]+(partition_sizes[i-1]-partition_offsets[i]);
                    MPI_Send(x,y,MPI_INT,i,0,MPI_COMM_WORLD);
                    //cout<<*x<<" send "<<i<<endl;
                }
                //cout << "after first send"<<endl;
                int index = 0;
                for(int i = 0; i < binslength; ++i){
                    while(bins[i].size()>0){
                        local[index] = *bins[i].front();
                        bins[i].pop_front();
                        index++;
                    }

                }
                //cout<<"end if rank 0"<<endl;
            }
            else{
                int y = 1;
                MPI_Recv(localCount,y,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                //cout <<"after first recv "<<*localCount<<endl;

            }
            //send particles back out to processes
            if(rank == 0){
                particle_t  *temp = tempP;
                //send particles
                int y;
                for(int i = 1; i < n_proc; ++i){
                   // cout<<partition_offsets[i]<<endl;
                    temp+=partition_offsets[i];
                    y = partition_sizes[i]+(partition_sizes[i-1]-partition_offsets[i]);
                    MPI_Send(temp,y,PARTICLE,i,0,MPI_COMM_WORLD);
                }
               // cout<<"after send particles"<<endl;
            }
            else{
                //cout<<"Local "<< *localCount <<endl;
                MPI_Recv(local,*localCount,PARTICLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                //cout<<"after 2nd rec"<<endl;
                //put particles in local bins
                particle_t p;
                for(int i =0; i < *localCount; ++i){
                    p= local[i];

                    int x = floor(p.x/myconst)+1;
                    int y = floor(p.y/myconst)+1;
                    bins[x*nn + y].push_front(&p);
                }
            }

            //cout<<*localCount<<" "<<rank<< endl;
            if (find_option(argc, argv, "-no") == -1) {
                MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
                MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
                MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
                //
                // Computing statistical data
                //
                if (rank == 0){
                    //
                    // Computing statistical data
                    //
                    if (rnavg) {
                        absavg +=  rdavg/rnavg;
                        nabsavg++;
                    }
                    if (rdmin < absmin) absmin = rdmin;
                }
            }
        }
    }
    //cout<<"at timer "<<endl;
    simulation_time = read_timer( ) - simulation_time;

    if (rank == 0) {
        printf( "n = %d, simulation time = %g seconds %d procs", n, simulation_time,n_proc);

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
            fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }


    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( particles );
    free(tempP);
    free(local);
    free( partition_offsets );
    free( partition_sizes );
    //cout<<"after free "<<rank<<endl;
    MPI_Finalize();
    return 0;
    if( fsave )
        fclose( fsave );
    MPI_Finalize( );
    return  0;

}
