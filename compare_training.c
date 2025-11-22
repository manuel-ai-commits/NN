#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};

int main(void) {
    srand(69);
    size_t stride = 3;
    size_t n = 4;
    
    Mat ti = {.rows = n, .cols = 2, .stride = stride, .es = td};
    Mat to = {.rows = n, .cols = 1, .stride = stride, .es = td + 2};
    
    size_t arch[] = {2,2,1};
    
    // Test with backprop
    NN nn1 = nn_alloc(arch, ARRAY_LEN(arch));
    NN g1 = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn1, 0, 1);
    
    // Test with finite diff  
    NN nn2 = nn_alloc(arch, ARRAY_LEN(arch));
    NN g2 = nn_alloc(arch, ARRAY_LEN(arch));
    // Copy same initial weights
    for (size_t l = 0; l < nn1.n_layers; ++l) {
        for (size_t i = 0; i < nn1.ws[l].rows; ++i) {
            for (size_t j = 0; j < nn1.ws[l].cols; ++j) {
                MAT_AT(nn2.ws[l], i, j) = MAT_AT(nn1.ws[l], i, j);
            }
        }
        for (size_t j = 0; j < nn1.bs[l].cols; ++j) {
            MAT_AT(nn2.bs[l], 0, j) = MAT_AT(nn1.bs[l], 0, j);
        }
    }
    
    float rate = 1e-1;
    size_t epochs = 1000;
    
    printf("Initial loss (both): %f\n", nn_loss(nn1, ti, to));
    
    for(size_t i = 0; i < epochs; ++i) {
        nn_backprop(nn1, g1, ti, to);
        nn_learn(nn1, g1, rate);
        
        nn_finite_diff(nn2, g2, 1e-1, ti, to);
        nn_learn(nn2, g2, rate);
        
        if (i % 100 == 0) {
            printf("Epoch %zu: Backprop loss=%f, FiniteDiff loss=%f\n", 
                   i, nn_loss(nn1, ti, to), nn_loss(nn2, ti, to));
        }
    }
    
    printf("\nFinal loss - Backprop: %f, FiniteDiff: %f\n", 
           nn_loss(nn1, ti, to), nn_loss(nn2, ti, to));
    
    return 0;
}
