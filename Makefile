.PHONY: help all opt clean distclean debug release reldebug test test-reldebug \
	fuzz-tools py py-clean py-rebuild py-debug py-release py-test py-typecheck py-lint py-e2e \
	regen-pyi dump-tests-json run-local-runner local-runner run-min-tests run-all-tests \
	check-builds check-installs static-analysis compile-commands clang-format

all: debug
opt: release

help:
	@printf '%s\n' \
		'Simjit make targets' \
		'' \
		'Core build:' \
		'  make debug                 Configure/build debug CLI' \
		'  make release               Configure/build release CLI' \
		'  make reldebug              Configure/build RelWithDebInfo library' \
		'  make test                  Build debug C++ test binaries' \
		'  make test-reldebug         Build RelWithDebInfo C++ test binaries' \
		'  make fuzz-tools            Build fuzz driver tools' \
		'' \
		'Python:' \
		'  make py                    Build local optimized Python extension' \
		'  make py-debug              Build local debug Python extension' \
		'  make py-test               Run Python unit tests' \
		'  make py-typecheck          Run BasedPyright on the Python package' \
		'  make py-lint               Run Ruff on the Python package' \
		'  make py-e2e                Run Python e2e tests' \
		'  make py-release            Build release wheel' \
		'' \
		'Test workflows:' \
		'  make run-min-tests         Run compiler test cases without runtime checking' \
		'  make run-all-tests         Run complete all-suite project workflow' \
		'  make check-builds          Check selected build configurations' \
		'  make check-installs        Build and check jit/toolkit install profiles' \
		'  make static-analysis       Build C++ library and Python extension with static analysis' \
		'  make compile-commands      Merge configured CMake compile databases' \
		'  make dump-tests-json       Dump full JSONL test bundle' \
		'  make run-local-runner      Run native runtime checks on full bundle' \
		'  make local-runner          Build thread-pooled native JSON bundle runner' \
		'' \
		'Cleanup:' \
		'  make clean                 Remove reports and test dumps' \
		'  make distclean             Remove build dirs, reports, and test dumps' \
		'  make py-clean              Remove Python dev build dir' \
		'' \
		'Common knobs:' \
		'  WORKERS=N                  Override parallel build workers' \
		'  GEN=ninja|make             Choose CMake generator wrapper behavior' \
		'  NO_LLVM=1                  Disable LLVM backend' \
		'  NO_ASMJIT=1                Disable AsmJit backend' \
		'  NO_CPP=1                   Disable C++ backend' \
		'  NO_X86=1 / NO_ARM=1        Disable one AsmJit architecture backend' \
		'  PY_LLVM=1 / PY_CPP=1       Enable optional Python inspection emitters' \
		'  NO_LTO=1                   Disable LTO' \
		'  LTO_MODE=THIN|FULL|AUTO    Choose LTO flavor' \
		'  ASAN=1 UBSAN=1 COVERAGE=1  Enable sanitizer or coverage flags' \
		'  SIMJIT_PYTHON=/path/python Override Python interpreter' \
		'' \
		'Optional benchmarks and Explorer commands: scripts/dev --help'

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD_DIR := $(ROOT)/build
DEBUG_BUILD_DIR := $(BUILD_DIR)/debug
RELEASE_BUILD_DIR := $(BUILD_DIR)/release
RELDEBUG_BUILD_DIR := $(BUILD_DIR)/reldebug
PYTHON_DIR := $(ROOT)/python
PYTHON_PACKAGE_SRC_DIR := $(PYTHON_DIR)/src/simjit
PYTHON_BUILD_DIR := $(BUILD_DIR)/python-dev
PYTHON_PACKAGE_BUILD_DIR := $(PYTHON_BUILD_DIR)/simjit
PYTHON_DIST_DIR := $(BUILD_DIR)/python-dist
TEST_DUMP_DIR := $(ROOT)/test-dump
COMPILE_COMMANDS_SCRIPT := $(ROOT)/scripts/merge-compile-commands.py
COMPILE_COMMANDS_BUILD_DIRS := $(DEBUG_BUILD_DIR) $(RELDEBUG_BUILD_DIR) $(RELEASE_BUILD_DIR) $(PYTHON_BUILD_DIR)

GEN ?= $(shell command -v ninja >/dev/null 2>&1 && printf ninja || printf make)
GENERATOR ?=
FORCE_COLOR ?=
ASAN_FLAG ?=
UBSAN_FLAG ?=
COVERAGE_FLAG ?=
LLVM_FLAG ?=
ASMJIT_FLAG ?=
CPP_FLAG ?=
SERIALIZATION_FLAG ?= -DSIMJIT_ENABLE_SERIALIZATION=ON
LTO_FLAG ?= -DSIMJIT_ENABLE_LTO=ON
NO_X86 ?= 
NO_ARM ?= 
WORKERS ?= $(shell command -v nproc >/dev/null 2>&1 && nproc || sysctl -n hw.ncpu 2>/dev/null || echo 4)
PYTHON_EXECUTABLE ?= $(shell if [ -n "$$SIMJIT_PYTHON" ]; then printf '%s\n' "$$SIMJIT_PYTHON"; elif [ -x "$(ROOT)/.venv/bin/python" ]; then printf '%s\n' "$(ROOT)/.venv/bin/python"; elif [ -x "$$HOME/.venv/bin/python" ]; then printf '%s\n' "$$HOME/.venv/bin/python"; else command -v python3; fi)
PY_PRESET ?= dev
PY_SCRIPT ?= $(ROOT)/scripts/py
PY_LLVM ?= 0
PY_CPP ?= 0
INSTALL_CHECK_DIR ?= $(BUILD_DIR)/install-check

ifeq ($(GEN),ninja)
	export CMAKE_GENERATOR := Ninja
	FORCE_COLOR=-DCMAKE_COLOR_DIAGNOSTICS=ON
endif
ifeq (${ASAN}, 1)
	ASAN_FLAG=-DSIMJIT_ENABLE_ASAN=ON
endif
ifeq (${UBSAN}, 1)
	UBSAN_FLAG=-DSIMJIT_ENABLE_UBSAN=ON
endif
ifeq (${COVERAGE}, 1)
	COVERAGE_FLAG=-DSIMJIT_ENABLE_COVERAGE=ON
endif
ifeq (${NO_LLVM},1)
	LLVM_FLAG=-DSIMJIT_ENABLE_LLVM=OFF
else
	LLVM_FLAG=-DSIMJIT_ENABLE_LLVM=ON
endif 
ifeq (${NO_ASMJIT},1)
	ASMJIT_FLAG=-DSIMJIT_ENABLE_ASMJIT=OFF
else
	ifeq (${NO_X86},1)
		ASMJIT_FLAG=-DSIMJIT_ENABLE_ASMJIT=ON -DSIMJIT_ENABLE_ASMJIT_ARM=ON -DSIMJIT_ENABLE_ASMJIT_X86=OFF
	else ifeq (${NO_ARM},1)
		ASMJIT_FLAG=-DSIMJIT_ENABLE_ASMJIT=ON -DSIMJIT_ENABLE_ASMJIT_ARM=OFF -DSIMJIT_ENABLE_ASMJIT_X86=ON
	else 
		ASMJIT_FLAG=-DSIMJIT_ENABLE_ASMJIT=ON -DSIMJIT_ENABLE_ASMJIT_ARM=ON -DSIMJIT_ENABLE_ASMJIT_X86=ON
	endif
endif 
ifeq (${NO_CPP},1)
	CPP_FLAG=-DSIMJIT_ENABLE_CPP=OFF
else
	CPP_FLAG=-DSIMJIT_ENABLE_CPP=ON
endif 
ifeq (${NO_LTO},1)
	LTO_FLAG=-DSIMJIT_ENABLE_LTO=OFF
endif
ifneq ($(LTO_MODE),)
	LTO_FLAG += -DSIMJIT_LTO_MODE=$(LTO_MODE)
endif
ALL_FLAGS:=$(GENERATOR) $(FORCE_COLOR) $(ASAN_FLAG) $(UBSAN_FLAG) $(COVERAGE_FLAG) $(LLVM_FLAG) $(ASMJIT_FLAG) $(CPP_FLAG) $(SERIALIZATION_FLAG) $(LTO_FLAG)
ifeq (${PY_LLVM},1)
	PYTHON_BACKEND_FLAGS += -DSIMJIT_PYTHON_ENABLE_LLVM=ON
else
	PYTHON_BACKEND_FLAGS += -DSIMJIT_PYTHON_ENABLE_LLVM=OFF
endif
ifeq (${PY_CPP},1)
	PYTHON_BACKEND_FLAGS += -DSIMJIT_PYTHON_ENABLE_CPP=ON
else
	PYTHON_BACKEND_FLAGS += -DSIMJIT_PYTHON_ENABLE_CPP=OFF
endif

clean:
	rm -rf "$(ROOT)"/report.* "$(TEST_DUMP_DIR)"

distclean:
	rm -rf "$(BUILD_DIR)" "$(ROOT)"/report.* "$(TEST_DUMP_DIR)"

compile-commands:
	"$(PYTHON_EXECUTABLE)" "$(COMPILE_COMMANDS_SCRIPT)" --output "$(ROOT)/compile_commands.json" $(COMPILE_COMMANDS_BUILD_DIRS)

debug: 
	cmake -S "$(ROOT)" --preset debug $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(DEBUG_BUILD_DIR)" --target simjit-cli --parallel $(WORKERS)

py:
	cmake -S "$(PYTHON_DIR)" --preset $(PY_PRESET) $(COVERAGE_FLAG) $(LTO_FLAG) $(PYTHON_BACKEND_FLAGS) -DPython_EXECUTABLE="$(PYTHON_EXECUTABLE)" -DSIMJIT_PYTHON_PACKAGE_DIR="$(PYTHON_PACKAGE_BUILD_DIR)"
	$(MAKE) --no-print-directory compile-commands
	rsync -a --delete --exclude '_simjit*.so' --exclude '_simjit*.pyd' "$(PYTHON_PACKAGE_SRC_DIR)/" "$(PYTHON_PACKAGE_BUILD_DIR)/"
	cmake --build "$(PYTHON_BUILD_DIR)" --target _simjit --parallel $(WORKERS)

py-clean:
	rm -rf "$(PYTHON_BUILD_DIR)"

py-rebuild: py-clean py

py-debug: PY_PRESET=debug
py-debug: py

py-release:
	rm -rf "$(PYTHON_DIST_DIR)" && \
	mkdir -p "$(PYTHON_DIST_DIR)" && \
	CMAKE_ARGS="$(PYTHON_BACKEND_FLAGS) $${CMAKE_ARGS:-}" \
	$(PYTHON_EXECUTABLE) -m build --wheel --no-isolation --outdir "$(PYTHON_DIST_DIR)" "$(PYTHON_DIR)"

py-test: py
	$(PY_SCRIPT) -m pytest -q "$(PYTHON_DIR)/tests/test.py"

py-typecheck:
	cd "$(ROOT)" && "$(PYTHON_EXECUTABLE)" -m basedpyright

py-lint:
	cd "$(ROOT)" && "$(PYTHON_EXECUTABLE)" -m ruff check python/src

py-e2e: py
	$(PY_SCRIPT) -m pytest -q "$(PYTHON_DIR)/tests/e2e_test.py"

regen-pyi: py
	tmp_dir=$$(mktemp -d /tmp/simjit-pyi.XXXXXX) && \
	trap 'rm -rf "$$tmp_dir"' EXIT && \
	$(PY_SCRIPT) -m pybind11_stubgen simjit._simjit --enum-class-locations CompilePolicy:simjit._simjit -o "$$tmp_dir" && \
	cp "$$tmp_dir/simjit/_simjit.pyi" "$(PYTHON_PACKAGE_SRC_DIR)/_simjit.pyi" && \
	$(PY_SCRIPT) scripts/add-license-headers.py "$(PYTHON_PACKAGE_SRC_DIR)/_simjit.pyi"

release: 
	cmake -S "$(ROOT)" --preset release $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(RELEASE_BUILD_DIR)" --target simjit-cli --parallel $(WORKERS)

reldebug: 
	cmake -S "$(ROOT)" --preset reldebug $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(RELDEBUG_BUILD_DIR)" --target simjit --parallel $(WORKERS)

test:
	cmake -S "$(ROOT)" --preset debug $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(DEBUG_BUILD_DIR)" --target test integration_test --parallel $(WORKERS)

local-runner:
	cmake -S "$(ROOT)" --preset debug $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(DEBUG_BUILD_DIR)" --target local_runner --parallel $(WORKERS)

dump-tests-json: test
	mkdir -p "$(TEST_DUMP_DIR)"
	"$(DEBUG_BUILD_DIR)/test" --suite=all --arch=native --mode=all --dump-json "$(TEST_DUMP_DIR)/tests.jsonl"

run-local-runner: local-runner dump-tests-json
	"$(DEBUG_BUILD_DIR)/local_runner" --file "$(TEST_DUMP_DIR)/tests.jsonl" --timings $(LOCAL_RUNNER_ARGS)

fuzz-tools:
	cmake -S "$(ROOT)" --preset debug-fuzz-tools $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(DEBUG_BUILD_DIR)" --target simjit-fuzz-tool --parallel $(WORKERS)

test-reldebug:
	cmake -S "$(ROOT)" --preset reldebug $(ALL_FLAGS)
	$(MAKE) --no-print-directory compile-commands
	cmake --build "$(RELDEBUG_BUILD_DIR)" --target test integration_test --parallel $(WORKERS)
	
run-min-tests: test
	"$(DEBUG_BUILD_DIR)/integration_test"
	"$(DEBUG_BUILD_DIR)/test" --suite=all --arch=all --mode=all --validate-serialization

run-all-tests:
	"$(ROOT)/tests/scripts/run-all-tests.sh"

check-builds:
	"$(ROOT)/tests/scripts/check-builds.sh"

check-installs:
	"$(ROOT)/tests/scripts/check-installs.sh" "$(INSTALL_CHECK_DIR)" "$(WORKERS)"

static-analysis: export ENABLE_STATIC_ANALYSIS=1
static-analysis: reldebug py

clang-format:
	find "$(ROOT)/src" \( -name \*.cpp -or -name \*.h \) -exec clang-format -i {} \;
	find "$(ROOT)/tests" \( -name \*.cpp -or -name \*.h \) -exec clang-format -i {} \;
	find "$(ROOT)/explorer" \( -name \*.cpp -or -name \*.h \) -exec clang-format -i {} \;
	find "$(ROOT)/python" \( -name \*.cpp -or -name \*.h \) -exec clang-format -i {} \;
	find "$(ROOT)/benchmarks" \( -name \*.cpp -or -name \*.h \) -exec clang-format -i {} \;
