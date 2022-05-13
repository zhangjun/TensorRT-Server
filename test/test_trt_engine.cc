#include <gtest/gtest.h>
#include "trt_engine.h"

TEST(trt, trt_engine) {
  size_t batch_size = 1;
  size_t workspace_size = 1 << 30;
  TRTEngine *engine = new TRTEngine(batch_size, workspace_size);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc,argv);
  RUN_ALL_TESTS();
  return 0;
}