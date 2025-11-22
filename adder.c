#include <stdint.h>
#include <time.h>
#include <raylib.h>

#define NN_IMPLEMENTATION
#include "nn.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "deps/sv/stb_image_write.h"

#define BITS 4

#define IMG_FACTOR 100

#define IMG_WIDTH  16*IMG_FACTOR
#define IMG_HEIGHT 9*IMG_FACTOR

uint32_t img_pixels[IMG_WIDTH* IMG_HEIGHT];

void nn_render(Olivec_Canvas img, NN nn) {

    size_t arch_count = nn.n_layers + 1;
    uint32_t neuron_color = 0xFF0000FF;
    uint32_t connection_color = 0xFF00FF00;
    uint32_t background_color = 0xFF181818;
    uint32_t low_color = 0x000000FF;    
    uint32_t high_color = 0x0000FF00; 
    olivec_fill(img, background_color);
    int neuron_radius = 25;
    int layer_border_vpad = 50;
    int layer_border_hpad = 50;
    int nn_width = img.width - 2*layer_border_hpad;
    int nn_height = img.height - 2*layer_border_vpad;

    int nn_x = img.width/2 -nn_width/2;
    int nn_y = img.height/2 - nn_height/2;
    int layer_hpad = nn_width/arch_count;
    for (size_t l = 0; l < arch_count; ++l){
        int layer_vpad1 = nn_height/nn.as[l].cols;

        for (size_t i = 0; i < nn.as[l].cols; ++i) { 
            int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
            int cy1 = nn_y + i*layer_vpad1 +layer_vpad1/2;
            
            if (l+1 <arch_count){
                int layer_vpad2 = nn_height/nn.as[l+1].cols;
                for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
                    int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = nn_y + j*layer_vpad2 +layer_vpad2/2;
                    uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.ws[l], i, j)));
                    uint32_t connection_color = 0xFF000000|low_color;
                    olivec_blend_color(&connection_color, (alpha<<(8*3))|high_color);
                    olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
                }
            }
            if (l>0){
                uint32_t alpha = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
                uint32_t neuron_color = 0xFF000000|low_color;
                olivec_blend_color(&neuron_color, (alpha<<(8*3))|high_color);
                olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
            } else {
                olivec_circle(img, cx1, cy1, neuron_radius, 0xFFAAAAAA);
            }
        }
    }
}

int main(void) {
    size_t n = (1<<BITS);
    size_t rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS + 1);
    
    for (size_t i = 0; i < ti.rows; ++i){
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x+y;
        for (size_t j = 0; j < BITS; ++j) {
            MAT_AT(ti, i , j)      = (x>>j)&1;
            MAT_AT(ti, i , j+BITS) = (y>>j)&1;
            MAT_AT(to,i,j)         = (z>>j)&1;
        } 
        MAT_AT(to, i ,BITS) = z >= n;

    } 

    size_t arch[] = {2*BITS,2*BITS + 1, BITS + 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    
    size_t epochs = 100*1000;
    float rate = 1;
    for (size_t i = 0; i < epochs; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        if(i%(10*1000) == 0){
            printf("loss = %f\n", nn_loss(nn, ti, to));
            Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
            nn_render(img, nn);
            char img_file_path[256];
            snprintf(img_file_path, sizeof(img_file_path), "out/xor-%03zu.png", i);
            if (!stbi_write_png(img_file_path, img.width, img.height, 4, img_pixels, img.stride * sizeof(uint32_t))){
                printf("ERROR: could not save the file %s\n", img_file_path);
                return 1;
            }
        }
    }

    size_t fails = 0;
    for (size_t x = 0; x < n; ++x) {
        for (size_t y = 0; y < n; ++y) {
            size_t z = x+y;     
            for (size_t j = 0; j < BITS; ++j) {
                MAT_AT(NN_INPUT(nn), 0 , j)      = (x>>j)&1;
                MAT_AT(NN_INPUT(nn), 0 , j+BITS) = (y>>j)&1;
        }
        nn_forward(nn);
        if (MAT_AT(NN_OUTPUT(nn), 0, BITS) >0.5f) {
            if (z<n) {printf("%zu + %zu = (OVERFLOW|%zu)\n", x, y, z); fails += 1;} 
        }
        else{
            size_t a = 0;
            for (size_t j = 0; j < BITS; ++j) {
                size_t bit = MAT_AT(NN_OUTPUT(nn),0,j) >= 0.5f;
                a |= bit<<j;
            }
            if (z!=a) {printf("%zu + %zu = (%zu<>%zu)\n", x, y, a, z); fails += 1;}
        }
        }
    }

    if (fails == 0) printf("OK\n");
    else printf("\nTotal number of fails = %zu\n", fails);

    return 0;
}