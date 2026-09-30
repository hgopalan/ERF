#ifndef FAST_LIBRARY_H
#define FAST_LIBRARY_H

// ERF stand-in for OpenFAST's C API header (modules/openfast-library/src/FAST_Library.h,
// OpenFAST 4.2.1, identical in 5.0.0, Apache-2.0). It declares only the entry points the ERF driver uses, with the
// same signatures, so ERF_OpenFASTDriver.cpp compiles unchanged against either this header or
// the real one. The implementation is ERF_OpenFASTStub.cpp: a rigid rotor with a uniform-Ct
// disk model, enough to exercise the coupling without an OpenFAST installation.

#include "ExternalInflow_Types.h"

#ifdef __cplusplus
#define EXTERNAL_ROUTINE extern "C"
#else
#define EXTERNAL_ROUTINE extern
#endif

EXTERNAL_ROUTINE void FAST_AllocateTurbines(int * iTurb, int *ErrStat, char *ErrMsg);
EXTERNAL_ROUTINE void FAST_DeallocateTurbines(int *ErrStat, char *ErrMsg);

EXTERNAL_ROUTINE void FAST_ExtInfw_Restart(int * iTurb, const char *CheckpointRootName, int *AbortErrLev, double * dt,
                                           int * NumBl, int * NumBlElem, int * NumTwrElem, int * n_t_global,
                                           ExtInfw_InputType_t* ExtInfw_Input, ExtInfw_OutputType_t* ExtInfw_Output,
                                           int *ErrStat, char *ErrMsg);
EXTERNAL_ROUTINE void FAST_ExtInfw_Init(int * iTurb, double *TMax, const char *InputFileName, int * TurbIDforName, char *OutFileRoot,
                                        int * NumActForcePtsBlade, int * NumActForcePtsTower, float * TurbinePosition, int *AbortErrLev,
                                        double * dtDriver, double * dt, int * InflowType, int * NumBl, int * NumBlElem, int * NumTwrElem, int * NodeClusterType,
                                        ExtInfw_InputType_t* ExtInfw_Input, ExtInfw_OutputType_t* ExtInfw_Output,
                                        int *ErrStat, char *ErrMsg);

EXTERNAL_ROUTINE void FAST_CFD_Solution0(int * iTurb, int *ErrStat, char *ErrMsg);
EXTERNAL_ROUTINE void FAST_CFD_Step(int * iTurb, int *ErrStat, char *ErrMsg);

EXTERNAL_ROUTINE void FAST_HubPosition(int * iTurb, float * absolute_position, float * rotation_veocity, double * orientation_dcm, int *ErrStat, char *ErrMsg);

EXTERNAL_ROUTINE void FAST_CreateCheckpoint(int * iTurb, const char *CheckpointRootName, int *ErrStat, char *ErrMsg);

// keep these synced with FAST_Library.f90
#define INTERFACE_STRING_LENGTH 1025

#define ErrID_None 0
#define ErrID_Info 1
#define ErrID_Warn 2
#define ErrID_Severe 3
#define ErrID_Fatal 4

#endif
