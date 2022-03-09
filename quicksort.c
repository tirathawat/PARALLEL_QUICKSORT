#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define INPUT_FILE argv[1]
#define OUTPUT_FILE argv[2]
#define MAX_THREADS atoi(argv[3])

double *read_file(char *path, int *size)
{
    int i;
    FILE *fp = fopen(path, "r");
    fscanf(fp, "%d", size);
    double *result = malloc(sizeof(double) * (*size));
    ;
    for (i = 0; i < *size; i++)
    {
        fscanf(fp, "%lf", &result[i]);
    }
    fclose(fp);
    return result;
}

void write_file(char *path, double *data, int size)
{
    int i;
    FILE *fp = fopen(path, "w");
    fprintf(fp, "%d\n", size);
    for (i = 0; i < size; i++)
    {
        fprintf(fp, "%lf\n", data[i]);
    }
    fclose(fp);
}

void calculate_size_of_chunks(int number_of_processes, double *arr, int arr_size, int *chunks, int *displacements)
{
    int i;

    int block_size = arr_size / number_of_processes;
    int remaining_size = arr_size % number_of_processes;

    int position = 0;

    for (i = 0; i < number_of_processes; i++)
    {
        chunks[i] = block_size;
        if (remaining_size > 0)
        {
            chunks[i]++;
            remaining_size--;
        }
        displacements[i] = position;
        position += chunks[i];
    }
}

void swap(double *arr, int i, int j)
{
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

int partition(double *arr, int low, int high)
{
    double pivot = arr[low];
    int i = --low;
    int j = ++high;

    while (1)
    {
        do
        {
            i++;
        } while (arr[i] < pivot);

        do
        {
            j--;
        } while (arr[j] > pivot);

        if (i >= j)
        {
            return j;
        }

        swap(arr, i, j);
    }
}

void quicksort(double *arr, int low, int high)
{
    if (low < high)
    {
        int pivot = partition(arr, low, high);
        quicksort(arr, low, pivot);
        quicksort(arr, pivot + 1, high);
    }
}

double *merge(double *arr1, double *arr2, int size1, int size2)
{
    int i = 0;
    int j = 0;
    int k = 0;

    int total_size = size1 + size2;
    double *merged_arr = malloc(sizeof(double) * total_size);

    while (i < size1 && j < size2)
    {
        if (arr1[i] < arr2[j])
        {
            merged_arr[k++] = arr1[i++];
        }
        else
        {
            merged_arr[k++] = arr2[j++];
        }
    }

    while (i < size1)
    {
        merged_arr[k++] = arr1[i++];
    }

    while (j < size2)
    {
        merged_arr[k++] = arr2[j++];
    }

    free(arr1);
    free(arr2);

    return merged_arr;
}

double *receive(double *data, int *chunks, int number_of_processes)
{
    double *sorted_arr = data;
    int i;
    int size = chunks[0];

    for (i = 1; i < number_of_processes; i += 1)
    {
        double *received_data = malloc(sizeof(double) * chunks[i]);
        MPI_Recv(received_data, chunks[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sorted_arr = merge(sorted_arr, received_data, size, chunks[i]);
        size += chunks[i];
    }

    return sorted_arr;
}

int main(int argc, char **argv)
{
    int number_of_processes, rank;
    double *arr;
    int arr_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunks[number_of_processes], displacements[number_of_processes];

    if (rank == 0)
    {
        arr = read_file(INPUT_FILE, &arr_size);
        calculate_size_of_chunks(number_of_processes, arr, arr_size, chunks, displacements);
    }

    int received_size;
    MPI_Scatter(chunks, 1, MPI_INT, &received_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double *data = malloc(sizeof(double) * received_size);
    MPI_Scatterv(arr, chunks, displacements, MPI_DOUBLE, data, received_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int pivot = partition(data, 0, received_size - 1);
    int left_pivot = partition(data, 0, pivot);
    int right_pivot = partition(data, pivot + 1, received_size - 1);

#pragma omp parallel num_threads(MAX_THREADS)
    {
#pragma omp sections
        {
#pragma omp section
            {

                quicksort(data, 0, left_pivot);
            }
#pragma omp section
            {
                quicksort(data, left_pivot + 1, pivot);
            }
#pragma omp section
            {
                quicksort(data, pivot + 1, right_pivot);
            }
#pragma omp section
            {
                quicksort(data, right_pivot + 1, received_size - 1);
            }
        }
    }

    if (rank == 0)
    {
        double *sorted_arr = receive(data, chunks, number_of_processes);
        write_file(OUTPUT_FILE, sorted_arr, arr_size);
    }
    else
    {
        MPI_Send(data, received_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}