// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// Isolated MPI bootstrap helper for NCCL. Only compiled when building the
// optional MPI-enabled NCCL module; MPI-free builds omit this file completely
// so no <mpi.h> or mpi_wrapper.h header is ever required.
#include "mpi_wrapper.h"

namespace jittor {

void nccl_mpi_bcast(void* buf, int count, int root) {
    MPI_CHECK(MPI_Bcast(buf, count, MPI_BYTE, root, MPI_COMM_WORLD));
}

} // jittor
