CC = cc
CFLAGS = -Wall -std=c99 $(shell pkg-config --cflags raylib)

LIBS = $(shell pkg-config --libs raylib)

main: gym.c
	$(CC) $(CFLAGS) gym.c -o game $(LIBS)

clean:
	rm -f game