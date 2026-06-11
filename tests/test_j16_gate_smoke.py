import sys, os
os.chdir("D:/Orb"); sys.path.insert(0, "D:/Orb")
from app.builds.stage_hooks import _spec_output_blocks_advance

# blocks on each marker, only for spec_gate
assert _spec_output_blocks_advance("spec_gate", "spec ok but [HUMAN_REQUIRED] choose auth provider") == "HUMAN_REQUIRED"
assert _spec_output_blocks_advance("spec_gate", "claim X is UNVERIFIED-EVIDENCE pending check") == "UNVERIFIED-EVIDENCE"
assert _spec_output_blocks_advance("spec_gate", "tag UNVERIFIED_EVIDENCE present") == "UNVERIFIED_EVIDENCE"
# clean spec advances
assert _spec_output_blocks_advance("spec_gate", "fully grounded spec, all evidence verified") is None
# other stages never blocked by this gate
assert _spec_output_blocks_advance("weaver", "HUMAN_REQUIRED in a weaver ramble") is None
assert _spec_output_blocks_advance("critical_pipeline", "UNVERIFIED-EVIDENCE") is None
# empty output never blocks
assert _spec_output_blocks_advance("spec_gate", "") is None
print("J16 GATE UNIT TEST PASS (7 paths)")

# compat adapter imports + signature sanity (no API call)
import inspect
from app.pipeline_v2.llm_compat import run_agentic_loop
sig = inspect.signature(run_agentic_loop)
assert list(sig.parameters)[:4] == ["prompt", "root_path", "language", "model"]
print("J16 COMPAT ADAPTER IMPORT OK")
