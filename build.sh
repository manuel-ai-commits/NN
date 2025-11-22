#!/bin/sh

set -x

cc -Wall -W -Wextra -O2 nn.c -lm
cc -Wall -W -Wextra -O2 dump_nn.c -lm
cc -Wall -W -Wextra -O2 nn1.c -lm
cc -Wall -W -Wextra -O2 adder.c -lm