CPP=clang++
CFLAG=-Wall -Wextra -g `pkg-config --cflags --libs opencv` #-Werror -O3 

all: compile exec clean

compile: mapping.o
	${CPP} -o mapping ${CFLAG} $^

%.o: src/%.cpp
	${CPP} -o $@ ${CFLAG} -c $<

exec:
	./mapping

clean:
	rm *.o mapping
