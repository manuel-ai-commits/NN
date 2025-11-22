// Header part
#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_MALLOC

/*
    Strucutre for matrix allocation    
*/
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es; // pointer to a continuos numbers of floats allocated with mat_alloc
} Mat;

typedef struct {
    size_t n_layers;
    Mat *ws;
    Mat *bs;
    Mat *as; //The amount of activation is count + 1
             // where 1 is the input x
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).n_layers]


#define MAT_AT(m, i, j) m.es[(i) * (m).stride + (j)] // Wrap everty parameter in paranthesis in case there is some complex expression like 'i+1'
                                                     // using stride to select only a given number of columns
#define ARRAY_LEN(xs) sizeof(xs)/sizeof(xs[0])


float float_rand(void);
float sigmoidf(float n);

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float n);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, const char* name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0) // thiQs basically define a alias to which the first argument is the one given
                                      // while the second is the the one given stringified
NN nn_alloc(size_t *neurons, size_t n_t_layers);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
void nn_rand(NN nn, float low, float high);
#define NN_PRINT(nn) nn_print(nn, #nn); // A macro for prinitng the whole structure of a neural network
void nn_forward(NN nn); 
float nn_loss(NN nn, Mat in, Mat out);
void nn_finite_diff(NN nn, NN g, float eps, Mat in, Mat out);
void nn_backprop(NN nn, NN g, Mat in, Mat out);
void nn_learn(NN nn, NN g, float rate);

#endif // NN_H_



// C part
#ifdef NN_IMPLEMENTATION
/* Generate random float */
float rand_float(void) {
    return(float) rand() / RAND_MAX;
}

float sigmoidf(float n) {
    return 1.f/(1.f+expf(-n));
}

// =================== MATRICES =================== //
/*
    Define a "malloc" cousin for out structure for memory allocation
 */
 Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(sizeof(m.es) * rows * cols);
    assert(m.es != NULL);

    return m;
} 

void mat_rand(Mat m, float low, float high) {
    for (size_t i = 0; i<m.rows; ++i) {
        for (size_t j = 0; j<m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }    
}

void mat_fill(Mat m, float n) {
    for (size_t i = 0; i<m.rows; ++i) {
        for (size_t j = 0; j<m.cols; ++j) {
            MAT_AT(m, i, j) = n;
        }
    }    
}



void mat_dot(Mat dst, Mat a, Mat b) {
    
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i<dst.rows; ++i) {
        for (size_t j = 0; j<dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k< n; ++k) {
                // Multiple each item of a row belonging to matrix 'a' with each item belonging to column 'b'
                // Sum the results to get the item in the new matrix
                MAT_AT(dst, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j);
            }
        }
    }    
    
}

Mat mat_row(Mat m, size_t row) {

    NN_ASSERT(row <= m.rows);

    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat dst, Mat src) {

    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for(size_t i = 0; i< dst.rows; ++i) {
        for(size_t j = 0; j< dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }

}

void mat_sum(Mat dst, Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i<a.rows; ++i) {
        for (size_t j = 0; j<a.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }    
    
}

void mat_sig(Mat m) {
    for (size_t i = 0; i<m.rows; ++i) {
        for (size_t j = 0; j<m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

/*
    A function that prints the strucutre of the matrix
    as represented in math
*/
void mat_print(Mat m, const char *name, size_t padding) {
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i<m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j<m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}


// =================== NN =================== //
/*
    Given an array with each neuron per layer, including the initial one
    and the number of layers in total, including the input.

    It returns a completely allocated neural network structure with weights
    and biases correctly formatted.
*/

NN nn_alloc(size_t *neurons, size_t n_t_layers) {

    NN_ASSERT(n_t_layers > 0);

    NN nn;
    nn.n_layers = n_t_layers - 1;

    nn.ws = malloc(sizeof(*nn.ws)*nn.n_layers);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = malloc(sizeof(*nn.bs)*nn.n_layers);
    NN_ASSERT(nn.bs != NULL);
    nn.as = malloc(sizeof(*nn.as)*(nn.n_layers + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, neurons[0]);
    for (size_t i = 1; i < n_t_layers; ++i) {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, neurons[i]);
        nn.bs[i - 1] = mat_alloc(1, neurons[i]);
        nn.as[i] = mat_alloc(1, neurons[i]);
    }

    return nn;
}


void nn_zero(NN nn) {
    for (size_t i = 0; i < nn.n_layers; ++i) {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.n_layers], 0);
}

void nn_print(NN nn, const char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i< nn.n_layers; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }

    printf("]\n");
}

void nn_rand(NN nn, float low, float high) {
    size_t nn_layers = nn.n_layers;

    for (size_t i = 1; i < nn_layers + 1; ++i) {
        mat_rand(nn.ws[i-1], low, high);
        mat_rand(nn.bs[i-1], low, high);
    }
}

void nn_forward(NN nn) {
    size_t n_layers = nn.n_layers;

    for (size_t i = 0; i < n_layers; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

float nn_loss(NN nn, Mat in, Mat out) {
    // input size must be equal to output size
    assert(in.rows == out.rows);
    // number of outputs must be equal to the predicted outputs
    assert(out.cols == NN_OUTPUT(nn).cols);

    size_t n = in.rows;
    float loss = 0;
    for (size_t i = 0; i<n; ++i) {
        Mat X = mat_row(in, i);
        Mat y = mat_row(out, i);
        mat_copy(NN_INPUT(nn), X);
        nn_forward(nn);
        size_t q = out.cols;
        for(size_t j = 0; j< q; ++j) { 
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            loss += d*d;
        }
    }
    return loss/=n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat in, Mat out) {
    // In this implementation, both weights and biases are updated by finite difference.
    // However, you might observe that after repeated epochs, the finite difference for weights stays at 0,
    // while the biases keep changing (if the loss surface is flat with respect to weights, gradients will be 0).
    float saved;

    float l_ = nn_loss(nn, in, out);

    for (size_t l = 0; l < nn.n_layers; ++l) {
        // Compute finite difference for weights
        for (size_t i = 0; i < nn.ws[l].rows; ++i) {
            for (size_t j = 0; j < nn.ws[l].cols; ++j) {
                saved = MAT_AT(nn.ws[l], i, j);
                MAT_AT(nn.ws[l], i, j) += eps;
                float grad = (nn_loss(nn, in, out) - l_) / eps;
                MAT_AT(g.ws[l], i, j) = grad;
                MAT_AT(nn.ws[l], i, j) = saved;
            }
        }

        // Compute finite difference for biases
        for (size_t i = 0; i < nn.bs[l].rows; ++i) {
            for (size_t j = 0; j < nn.bs[l].cols; ++j) {
                saved = MAT_AT(nn.bs[l], i, j);
                MAT_AT(nn.bs[l], i, j) += eps;
                float grad = (nn_loss(nn, in, out) - l_) / eps;
                MAT_AT(g.bs[l], i, j) = grad;
                MAT_AT(nn.bs[l], i, j) = saved;
            }
        }
    }
}

void nn_backprop(NN nn, NN g, Mat in, Mat out) {
    NN_ASSERT(in.rows == in.rows);
    size_t n = in.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == out.cols);

    nn_zero(g); 
    // 'i' - Current sample 
    // 'l' - Current layer
    // 'j' - Current neuron activation
    // 'k' - Previous neuron activation
    
    for (size_t i = 0; i < n; ++i) {
        mat_copy(NN_INPUT(nn), mat_row(in, i));
        nn_forward(nn);
        
        for (size_t j = 0; j <= nn.n_layers; ++j) {
            mat_fill(g.as[j], 0);
        }

        for (size_t j= 0; j < out.cols ; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) = 2*(MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(out, i, j));
        }

        for (size_t l = nn.n_layers; l > 0 ; --l) {
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l-1], 0, j) += da*a*(1 - a);
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    // 'j' - weight matrix column
                    // 'k' - weight matrix row
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += da*a*(1 - a)*pa;
                    MAT_AT(g.as[l-1], 0, k) += da*a*(1 - a)*w;
                }
            }
        }
    }
    for (size_t i = 0; i< g.n_layers; ++i) {
        for (size_t j = 0; j< g.ws[i].rows; ++j) {
            for (size_t k = 0; k< g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }

        for (size_t k = 0; k< g.bs[i].cols; ++k) {
            MAT_AT(g.bs[i], 0, k) /= n;
        }
        
    }
}



void nn_learn(NN nn, NN g, float rate) {
    // The learning step is identical for both weights and biases.
    // If the gradients (g.ws) for weights remain 0 over epochs, weights do not change,
    // but if g.bs is nonzero, the biases keep updating.
    for (size_t l = 0; l < nn.n_layers; ++l) {
        for (size_t i = 0; i < nn.ws[l].rows; ++i) {
            for (size_t j = 0; j < nn.ws[l].cols; ++j) {
                MAT_AT(nn.ws[l], i, j) -= MAT_AT(g.ws[l], i, j) * rate;
            }
        }
        for (size_t i = 0; i < nn.bs[l].rows; ++i) {
            for (size_t j = 0; j < nn.bs[l].cols; ++j) {
                MAT_AT(nn.bs[l], i, j) -= MAT_AT(g.bs[l], i, j) * rate;
            }
        }
    }
}

#endif // NN_IMPLEMENTATION