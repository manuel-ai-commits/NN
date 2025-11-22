#include <stdio.h>
#include <raylib.h>

int main(void){

    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "urmom");

    SetTargetFPS(60);

    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        BeginDrawing();
            ClearBackground(RAYWHITE);
            DrawCircle(screenWidth/2, screenHeight/2, 100, RED);
        EndDrawing();
    }

    CloseWindow();              // Close window and OpenGL context
    return 0;
}
