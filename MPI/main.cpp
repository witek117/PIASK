#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

double polynominal(double x){
    return 5*pow(x,4) + 4*pow(x,3) + x - 10*pow(x,2);
}

void print_domain(const double* domain, int length){
    printf("Domain:\n");
    for(int i = 0; i < length ; i++){
        printf("%d: %.3f\n", i , *domain);
        domain++;
    }
}

double calculateTrapezoidField(double a, double b, double h) {
    return ((a + b) * h / 2);
}

void compute(const double* sub_domain, int sub_domain_length, void* params, double* local_integral){
    auto *dx = reinterpret_cast<double*>(params);
    double sum = 0.0f;
    double y1, y2;
    y1 = polynominal(sub_domain[0]);

    for(int i = 1; i < sub_domain_length; i++){
        y2 = polynominal(sub_domain[i]);
        sum += calculateTrapezoidField(y1, y2, *dx);
        y1 = y2;
    }
    *local_integral = sum;
}

double* get_full_domain(double startValue, double stopValue, double dx, int* length){
    int len =  (int)((stopValue - startValue) / dx) + 1;
    (*length) = len;
    auto* domain = (double*)malloc(len * sizeof(double));

    double value = startValue;

    for(int i = 0; i < len; i++) {
        domain[i] = value;
        value += dx;
    }

    return domain;
}


int main(int argc, char** argv) {
    double dx = 0.001;
    double start = 0;
    double stop = 2;
    int myrank;
    int size;
    int root = 1;
    int full_domain_length;
    int sub_domain_length;
    double global_integral;
    double local_integral;

    double* full_domain;
    double* sub_domain;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    root = 0;

    if (myrank == root) {
        full_domain = get_full_domain(start, stop, dx ,&full_domain_length);
    }

    MPI_Bcast(&full_domain_length, 1, MPI_INT, root, MPI_COMM_WORLD);

    sub_domain_length = full_domain_length / size;

    sub_domain = (double*) malloc (sub_domain_length * sizeof(double));

    MPI_Scatter(full_domain, sub_domain_length, MPI_DOUBLE, sub_domain, sub_domain_length, MPI_DOUBLE, root, MPI_COMM_WORLD);

    compute(sub_domain, sub_domain_length, &dx, &local_integral);

    MPI_Reduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    MPI_Finalize();

    if (myrank == root) {
        // sum integral between sub domains
        for(int i =0; i < (size - 1); i++) {
            int index = (i * sub_domain_length) + sub_domain_length;
            global_integral += calculateTrapezoidField(polynominal(full_domain[index - 1]), polynominal(full_domain[index]), dx);
        }

        int rest = full_domain_length % size;
        int startRestIndex = (sub_domain_length * size) - 1;
        int stopRestIndex = startRestIndex + rest;

        double y1 = polynominal(full_domain[startRestIndex]);
        double y2;

        // sum integral after last sub domain, if exists
        for(int i = startRestIndex; i < stopRestIndex; i++) {
            y2 = polynominal(full_domain[i+1]);
            global_integral += calculateTrapezoidField(y1, y2, dx);
            y1 = y2;
        }

        printf("Global integral: %.3f\n", global_integral);
    }

    return 0;
}