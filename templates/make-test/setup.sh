#!/bin/bash
. ../../setup.sh
spackLoadUnique >>>PACKAGE<<<
export E4S_TEST_SOURCE=>>>SOURCEFILE.EXT<<<
export E4S_TEST_FLAGS=>>>FLAGS<<<
export E4S_TEST_LIBS=>>>LIBS<<<
