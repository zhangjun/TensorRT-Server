#include "trt_engine.h"

int main() {
  size_t batch_size = 1;
  size_t workspace_size = 1 << 30;
  TRTEngine *engine = new TRTEngine(batch_size, workspace_size);
}
