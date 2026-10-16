namespace amrex {

const char* buildInfoGetBuildDate() {

  static const char BUILD_DATE[] = __DATE__ " "  __TIME__;
  return BUILD_DATE;
}

const char* buildInfoGetBuildDir() {

  static const char BUILD_DIR[] = "/home/runner/work/ERF/ERF/build/Exec";
  return BUILD_DIR;
}

const char* buildInfoGetBuildMachine() {

  static const char BUILD_MACHINE[] = "Linux runnervmmprz5 6.17.0-1022-azure #22-Ubuntu SMP Mon Jul 27 17:24:03 UTC 2026 x86_64";
  return BUILD_MACHINE;
}

const char* buildInfoGetAMReXDir() {

  static const char AMREX_DIR[] = "/home/runner/work/ERF/ERF/Submodules/AMReX";
  return AMREX_DIR;
}

const char* buildInfoGetComp() {

  static const char COMP[] = "GNU";
  return COMP;
}

const char* buildInfoGetCompVersion() {

  static const char COMP_VERSION[] = "13.3.0";
  return COMP_VERSION;
}

// deprecated
const char* buildInfoGetFcomp() {

  static const char FCOMP[] = "";
  return FCOMP;
}

// deprecated
const char* buildInfoGetFcompVersion() {

  static const char FCOMP_VERSION[] = "";
  return FCOMP_VERSION;
}

const char* buildInfoGetCXXName() {

  static const char CXX_comp_name[] = "/usr/bin/c++";
  return CXX_comp_name;
}

const char* buildInfoGetFName() {

  static const char F_comp_name[] = "";
  return F_comp_name;
}

const char* buildInfoGetCXXFlags() {

  static const char CXX_flags[] = "-DERF_USE_MOISTURE -DERF_WDM6_F32_LITERALS -DERF_USE_FFT -DERF_HAS_SHOC_FAMILY -DERF_USE_MOVING_BODIES -DERF_USE_OPENFAST -I/home/runner/work/ERF/ERF/Source/PBL/Shoc -I/home/runner/work/ERF/ERF/Source/MovingBodies -I/home/runner/work/ERF/ERF/Source/MovingBodies/Core -I/home/runner/work/ERF/ERF/Source/MovingBodies/Motion -I/home/runner/work/ERF/ERF/Source/MovingBodies/ForceModels -I/home/runner/work/ERF/ERF/Source/MovingBodies/OpenFAST -I/home/runner/work/ERF/ERF/Source/MovingBodies/OpenFAST/Stub -O2 -g -DNDEBUG";
  return CXX_flags;
}

const char* buildInfoGetFFlags() {

  static const char F_flags[] = "";
  return F_flags;
}

const char* buildInfoGetLinkFlags() {

  static const char link_flags[] = "";
  return link_flags;
}

const char* buildInfoGetLibraries() {

  static const char libraries[] = "";
  return libraries;
}

const char* buildInfoGetAux(int i) {

  //static const char AUX1[] = "${AUX[1]}";
  
  static const char EMPT[] = "";

  switch(i)
  {
    
    default: return EMPT;
  }
}

int buildInfoGetNumModules() {
  // int const num_modules = X;
  int const num_modules = 0;
  return num_modules;
}

const char* buildInfoGetModuleName(int i) {

  //static const char MNAME1[] = "${MNAME[1]}";
  
  static const char EMPT[] = "";

  switch(i)
  {
    
    default: return EMPT;
  }
}

const char* buildInfoGetModuleVal(int i) {

  //static const char MVAL1[] = "${MVAL[1]}";
  
  static const char EMPT[] = "";

  switch(i)
  {
    
    default: return EMPT;
  }
}

const char* buildInfoGetGitHash(int i) {

  //static const char HASH1[] = "${GIT[1]}";
  static const char HASH1[] = "83bd7008f2c0-dirty";
  static const char HASH2[] = "26.09";
  static const char EMPT[] = "";

  switch(i)
  {
    case 1: return HASH1;
    case 2: return HASH2;
    default: return EMPT;
  }
}

const char* buildInfoGetBuildGitHash() {

  //static const char HASH[] = "${GIT}";
  static const char HASH[] = "";

  return HASH;
}

const char* buildInfoGetBuildGitName() {

  //static const char NAME[] = "";
  static const char NAME[] = "";

  return NAME;
}

}
