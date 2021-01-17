#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#define PRECISION 0.01
#define POINTS_NUMBER (4.0 / PRECISION)

void compute(double* sub_domain, int sub_domain_length, double* local_min){
    double y;
    double y_min = LLONG_MAX;
    double x = 0;
    double x_min = 0;
    int i = 0;

    for(i = 0; i < sub_domain_length; i++){
        x = sub_domain[i];
        y = pow((x-5), 3.0) - pow((x - 4), 2.0) + 1;
        if (y_min > y){
            y_min = y;
            x_min = x;
        }
    }

    *local_min = y_min;

    // printf("minimum point: (x:%f, y:%f) in <%.2f, %.2f>\n", x_min, y_min, sub_domain[0], sub_domain[sub_domain_length - 1]);
}

void print_domain(double* domain, int length){
    int i = 0;
    for(i = 0; i < length; i++){
        printf("%d: %.3f\n", i , *domain);
        domain++;
    }
}

double * get_full_domain(int* length){
    int i = 0;

    *length = POINTS_NUMBER;
    double* domain = (double*)malloc((*length) * sizeof(double));

    domain[0] = 4.0;
    for(i = 1; i < (*length); i++){
        domain[i] = domain[i-1] + PRECISION;
    }

    // printf("%.3f \n", domain[0]);
    return domain;
    // print_domain(*domain - (*length), *length);
}


int main(int argc, char** argv) {
    int myrank;
    int size;
    int root = 1;
    int full_domain_length;
    int sub_domain_length;
    double global_min;
    double local_min;

    double* full_domain;
    double* sub_domain;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    root = 0;

    if (myrank == root) {
        // full_domain = (double*)malloc(400 * sizeof(double));
        full_domain = get_full_domain(&full_domain_length);
        // printf("%.3f \n", full_domain[399]);

        // print_domain(full_domain, full_domain_length);
    }

    MPI_Bcast(&full_domain_length, 1, MPI_INT, root, MPI_COMM_WORLD);

    sub_domain_length = full_domain_length / size;

    // printf("Sub_domain_len %d    full %d\n", sub_domain_length, full_domain_length);

    sub_domain = (double*) malloc (sub_domain_length * sizeof(double));

    MPI_Scatter(full_domain, sub_domain_length, MPI_DOUBLE, sub_domain, sub_domain_length, MPI_DOUBLE, root, MPI_COMM_WORLD);

    compute(sub_domain, sub_domain_length, &local_min);

    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);

    //MPI_Gather(sub_domain, sub_domain_length, MPI_DOUBLE, full_domain, full_domain_length, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Finalize();

    if (myrank == root) {
        printf("Global min: %.3f\n", global_min);
    }



    return 0;
}