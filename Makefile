# Make sure you have BLIS installed in your home directory (or some other BLAS library)
BLAS_LIB  := $(HOME)/blis/lib/libblis.a
BLAS_INC  := $(HOME)/blis/include/blis

# indicate how the object files are to be created
CC         := gcc-9
LINKER     := $(CC)
OPT_FLAGS  := -O3 -march=native
OPM_FLAGS  := -fopenmp
STD_FLAGS  := -std=c99  -D_POSIX_C_SOURCE=200809L
CFLAGS     := $(OPT_FLAGS) $(OPM_FLAGS) -I$(BLAS_INC) $(STD_FLAGS) -m64 -mavx2
FFLAGS     := $(CFLAGS) 

LDFLAGS    := -lpthread -m64 -lm -fopenmp

# grab object files for util
UTIL_DIR   := ./util
UTIL       := $(wildcard $(UTIL_DIR)/*.c)
UTIL_OBJS  := $(patsubst %.c,%.o,$(UTIL))

# grab object files for driver
DRIVER_DIR   := .
DRIVER       := $(wildcard $(DRIVER_DIR)/*.c)
DRIVER_OBJS  := $(patsubst %.c,%.o,$(DRIVER))
        
# grab all objects files
TEST_OBJS := $(DRIVER_OBJS) $(UTIL_OBJS)

# grab the header file
HEADERS := driver.h

# -------------- Set parameters ----------

# set the range of experiments to be performed
# smallest ~ 100
# biggest ~ 2000
PDEF_ST  := -DP_BEGIN=48 \
            -DP_END=2400 \
            -DP_INC=48

# grab the number of the group
GEMM     := $(wildcard $(DRIVER_DIR)/gemm*.c)
sbGEMM   := $(subst .c,,$(GEMM))
NUM      := $(subst ./gemm_,,$(sbGEMM))
GNUM     := -DGNUM=$(NUM)


# ------------- Set intermediates --------

.INTERMEDIATE: $(TEST_OBJS)

# -------------- Target(s) ---------------

all: driver.x

# compilation rule
%.o: %.c Makefile $(HEADERS)
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) $(PDEF_ST) $(GNUM) -c $< -o $@

# linker rule
driver.x: $(TEST_OBJS)
	@echo "Creating $@" 
	@$(LINKER) $(TEST_OBJS) $(BLAS_LIB) -o $@ $(LDFLAGS)

# clean rule                                                               
clean:
	@echo "Cleaning Object Files"
	@rm -f $(TEST_OBJS) *.x
