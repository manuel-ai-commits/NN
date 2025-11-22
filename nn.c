#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Mat o0;
    Mat w1, b1, o1;
    Mat w2, b2, o2;
} Xor_arch;

Xor_arch Xor_alloc() {
    Xor_arch m;
    // input Xor_arch
    m.o0 = mat_alloc(1,2);
    
    // First layer!
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    // matrix resulting from the first layer
    m.o1 = mat_alloc(1,2);


    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);

    // matrix resulting from the second layer
    m.o2 = mat_alloc(1,1);    

    return m;
}

void forward_xor(Xor_arch m) {
    mat_dot(m.o1, m.o1, m.w1);
    mat_sum(m.o1, m.b1);
    mat_sig(m.o1);

    mat_dot(m.o2, m.o1, m.w2);
    mat_sum(m.o2, m.b2);
    mat_sig(m.o2);
}


float loss(Xor_arch m, Mat ti, Mat to) {
    
    // input size must be equal to output size
    assert(ti.rows == to.rows);
    // number of outputs must be equal to the predicted outputs
    assert(to.cols == m.o2.cols );
    size_t n = ti.rows;
    
    float l = 0.0;
    for(size_t i = 0; i<n; ++i){
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        
        mat_copy(m.o0, x);
        forward_xor(m);
        
        size_t q = to.cols;
        for(size_t j = 0; j < q; ++j) {
            float d = MAT_AT(m.o2, 0, j) - MAT_AT(y, 0, j);
            l += d*d;
        }
    }

    return l /= n;
}

void finite_diff(Xor_arch m, Xor_arch g, float eps, Mat ti, Mat to) {

    float saved;
    float l = loss(m, ti, to);

    for (size_t i = 0; i < m.w1.rows; ++i) {
        for (size_t j = 0; j < m.w1.cols; ++j) {
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (loss(m, ti, to)- l)/eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; ++i) {
        for (size_t j = 0; j < m.b1.cols; ++j) {
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (loss(m, ti, to)- l)/eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.w2.rows; ++i) {
        for (size_t j = 0; j < m.w2.cols; ++j) {
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (loss(m, ti, to)- l)/eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.rows; ++i) {
        for (size_t j = 0; j < m.b2.cols; ++j) {
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (loss(m, ti, to)- l)/eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }
}

void learn(Xor_arch m, Xor_arch g, float rate) { 

    for (size_t i = 0; i < m.w1.rows; ++i) {
        for (size_t j = 0; j < m.w1.cols; ++j) {
            MAT_AT(m.w1, i, j) -= MAT_AT(g.w1, i, j)*rate;
        }
    }

    for (size_t i = 0; i < m.b1.rows; ++i) {
        for (size_t j = 0; j < m.b1.cols; ++j) {
            MAT_AT(m.b1, i, j) -=  MAT_AT(g.b1, i, j)*rate;
        }
    }

    for (size_t i = 0; i < m.w2.rows; ++i) {
        for (size_t j = 0; j < m.w2.cols; ++j) {
            MAT_AT(m.w2, i, j) -= MAT_AT(g.w2, i, j)*rate;
        }
    }

    for (size_t i = 0; i < m.b2.rows; ++i) {
        for (size_t j = 0; j < m.b2.cols; ++j) {
            MAT_AT(m.b2, i, j) -= MAT_AT(g.b2, i, j)*rate;
        }
    }
}


float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};


int main(void) {
    srand(time(0));
    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/3;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td,
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
    };

    Xor_arch m = Xor_alloc();
    Xor_arch g = Xor_alloc();
     
    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("loss = %f\n", loss(m, ti, to));
    finite_diff(m, g, eps, ti, to);
    learn(m, g, rate);
    printf("loss = %f\n", loss(m, ti, to));


    return 0;
}