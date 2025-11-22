// Gym is a GUI app that trains your NN on the data you give it.
// It will spit out the final file that can be loaded up with nn.h
// and used for your application
#include <raylib.h>
#include <stdio.h>

#define IMG_FACTOR 80
#define IMG_WIDTH  (16*(IMG_FACTOR))
#define IMG_HEIGHT (9*(IMG_FACTOR))

int main(int argc, char **argv){

    int buffer_len = 0;
    unsigned char *buffer = LoadFileData("adder.arch", &buffer_len);
    // If written like this, it shows you the file
    fwrite(buffer, buffer_len, 1, stdout);


    // InitWindow(IMG_WIDTH, IMG_HEIGHT, "adder");
    // SetTargetFPS(60);

    // while(!WindowShouldClose()){
    //     BeginDrawing();
    //     nn_render_raylib(nn);
    //     EndDrawing();
    // }
}